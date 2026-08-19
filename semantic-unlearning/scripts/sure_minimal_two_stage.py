#!/usr/bin/env python3
"""Minimal shared two-stage SURE-LM learner for MCF and ZsRE.

The architecture is intentionally narrow:

* frozen Base transformer and frozen Base input embeddings;
* exactly 50 direct sensitive forget requests;
* no benchmark retain examples and no replacement/reference targets;
* one fixed external Wikipedia final-hidden second moment;
* a row-specific rank-2 contrastive generalized-eigen basis;
* Stage 1: sensitive GA plus same-prompt non-sensitive GD;
* Stage 2: the same objective on failed direct token cases only, rank 2;
* smallest directly successful scale; no retain/norm/KL selection guard.

Official paraphrase/generalization, specificity, retain, exact retain-KL, and
PPL metrics are post-training audits and cannot affect this learner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_context_projection as context
import sure_shared_suppression as shared
import build_sure_wikipedia_stats as wikipedia


METHOD = "SURE-LM-minimal-Wikipedia-rank2-two-stage"
PROTOCOL = "sure_minimal_wikipedia_two_stage_v1"
FIXED_CONTRASTIVE_RANK = 2
DEFAULT_CANDIDATE_SCALES = (
    "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,"
    ".046875,.03125,.015625,.0078125,0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
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

    parser.add_argument("--stage1-steps", type=int, default=600)
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--stage1-lr", type=float, default=5e-3)
    parser.add_argument("--stage2-steps", type=int, default=500)
    parser.add_argument("--stage2-batch-size", type=int, default=8)
    parser.add_argument("--stage2-lr", type=float, default=5e-3)
    parser.add_argument("--stage2-check-every", type=int, default=25)
    parser.add_argument("--cache-batch-size", type=int, default=8)

    parser.add_argument("--ga-weight", type=float, default=2.0)
    parser.add_argument("--gd-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-eps", type=float, default=1e-3)
    parser.add_argument("--constraint-margin", type=float, default=0.05)
    parser.add_argument("--min-sensitive-nll-increase", type=float, default=4.0)
    parser.add_argument("--candidate-scales", default=DEFAULT_CANDIDATE_SCALES)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> List[float]:
    if args.forget_num <= 0:
        raise ValueError("forget-num must be positive")
    if args.utility_sample_size != wikipedia.DEFAULT_SAMPLE_SIZE:
        raise ValueError(
            "The locked minimal architecture requires exactly "
            f"a {wikipedia.DEFAULT_SAMPLE_SIZE}-document Wikipedia sample request"
        )
    positive = {
        "stage1_steps": args.stage1_steps,
        "stage1_batch_size": args.stage1_batch_size,
        "stage1_lr": args.stage1_lr,
        "stage2_steps": args.stage2_steps,
        "stage2_batch_size": args.stage2_batch_size,
        "stage2_lr": args.stage2_lr,
        "stage2_check_every": args.stage2_check_every,
        "cache_batch_size": args.cache_batch_size,
        "ga_weight": args.ga_weight,
        "contrastive_eps": args.contrastive_eps,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.gd_weight) or args.gd_weight < 0:
        raise ValueError("gd-weight must be finite and non-negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError("grad-clip must be finite and non-negative")
    if not math.isfinite(args.constraint_margin):
        raise ValueError("constraint-margin must be finite")
    if (
        not math.isfinite(args.min_sensitive_nll_increase)
        or args.min_sensitive_nll_increase < 0
    ):
        raise ValueError("min-sensitive-nll-increase must be finite and non-negative")
    scales = core.parse_scales(args.candidate_scales)
    if 0.0 not in scales or 1.0 not in scales:
        raise ValueError("candidate-scales must include both 0 and 1")
    return scales


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    expected_hash = manifest.get("training_visible_forget_sha256")
    if expected_hash != sha256_bytes(forget_bytes):
        raise RuntimeError("training-visible forget hash mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != args.forget_num:
        raise RuntimeError("split manifest forget count mismatch")
    if int(sampling.get("benchmark_retain_train_num", -1)) != 0:
        raise RuntimeError("minimal SURE forbids benchmark retain training examples")
    if "training_visible_retain" in manifest:
        raise RuntimeError("split manifest unexpectedly exposes a retain-training file")

    sensitive_field = core.sensitive_answer_field(args.dataset)
    forbidden_field = "target_true" if args.dataset == "mcf" else "target_new"
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
        if forbidden_field in rewrite:
            raise RuntimeError(
                f"forget record {position} exposes forbidden {forbidden_field}"
            )
    return forget_records, manifest


def load_utility_cache(
    path: Path,
    *,
    expected_sample_size: int,
    expected_hidden_size: int,
    expected_model_probe: str,
    expected_tokenizer_probe: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("utility cache must be a mapping")
    moment = payload.get("second_moment")
    metadata = payload.get("metadata")
    if not isinstance(moment, torch.Tensor) or not isinstance(metadata, Mapping):
        raise ValueError("utility cache lacks second_moment/metadata")
    metadata = dict(metadata)
    if metadata.get("protocol") != wikipedia.UTILITY_PROTOCOL:
        raise ValueError("utility cache protocol mismatch")
    if int(metadata.get("requested_document_sample_size", -1)) != expected_sample_size:
        raise ValueError("utility cache requested document sample size mismatch")
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
    if moment.shape != (expected_hidden_size, expected_hidden_size):
        raise ValueError("utility second moment has the wrong shape")
    if not torch.isfinite(moment).all():
        raise ValueError("utility second moment contains non-finite values")
    if not torch.allclose(moment, moment.transpose(0, 1), rtol=1e-4, atol=1e-5):
        raise ValueError("utility second moment is not symmetric")
    expected_sha = metadata.get("second_moment_sha256")
    if expected_sha and wikipedia.sha256_tensor(moment) != expected_sha:
        raise ValueError("utility second-moment checksum mismatch")
    return moment, metadata


@torch.no_grad()
def build_contrastive_bases_from_second_moment(
    forget_hidden: torch.Tensor,
    forget_tids: torch.Tensor,
    utility_second_moment: torch.Tensor,
    *,
    requested_ids: Sequence[int],
    rank_cap: int = FIXED_CONTRASTIVE_RANK,
    relative_eps: float,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    if rank_cap != FIXED_CONTRASTIVE_RANK:
        raise ValueError(
            f"Minimal SURE fixes both stages at rank {FIXED_CONTRASTIVE_RANK}"
        )
    if relative_eps <= 0:
        raise ValueError("contrastive epsilon must be positive")
    if forget_hidden.ndim != 2 or forget_hidden.shape[0] != forget_tids.shape[0]:
        raise ValueError("forget hidden states and token ids do not align")
    if utility_second_moment.shape != (
        forget_hidden.shape[1],
        forget_hidden.shape[1],
    ):
        raise ValueError("utility second moment and forget hidden size differ")

    utility = utility_second_moment.to(device=forget_hidden.device, dtype=torch.float32)
    token_ids = forget_tids.to(device=forget_hidden.device)
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for token_id in [int(x) for x in requested_ids]:
        rows = forget_hidden[token_ids.eq(token_id)].float()
        if rows.numel() == 0:
            raise RuntimeError(
                f"Sensitive token {token_id} has no direct forget hidden states"
            )
        forget_span = core.orthonormal_row_basis(rows, max_rank=None).float()
        if forget_span.ndim != 2 or forget_span.shape[0] == 0:
            raise RuntimeError(
                f"Sensitive token {token_id} has zero forget-context rank"
            )

        projected_forget = rows @ forget_span.transpose(0, 1)
        forget_cov = (projected_forget.transpose(0, 1) @ projected_forget) / float(
            rows.shape[0]
        )
        utility_cov = forget_span @ utility @ forget_span.transpose(0, 1)
        utility_cov = 0.5 * (utility_cov + utility_cov.transpose(0, 1))

        dimension = int(forget_span.shape[0])
        average_utility = (
            (torch.trace(utility_cov) / float(dimension)).abs().clamp_min(1e-6)
        )
        ridge = float(relative_eps) * average_utility
        minimum_utility_eigenvalue = torch.linalg.eigvalsh(utility_cov).min()
        psd_correction = (-minimum_utility_eigenvalue).clamp_min(0.0)
        diagonal_shift = ridge + psd_correction
        regularized = utility_cov + diagonal_shift * torch.eye(
            dimension,
            device=utility_cov.device,
            dtype=torch.float32,
        )
        chol = torch.linalg.cholesky(regularized)
        left = torch.linalg.solve_triangular(chol, forget_cov, upper=False)
        whitened = torch.linalg.solve_triangular(
            chol, left.transpose(0, 1), upper=False
        ).transpose(0, 1)
        whitened = 0.5 * (whitened + whitened.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
        order = torch.argsort(eigenvalues, descending=True)
        take = min(FIXED_CONTRASTIVE_RANK, dimension)
        whitened_directions = eigenvectors[:, order[:take]]
        coordinates = torch.linalg.solve_triangular(
            chol.transpose(0, 1), whitened_directions, upper=True
        )
        raw_directions = coordinates.transpose(0, 1) @ forget_span
        basis = core.orthonormal_row_basis(raw_directions, max_rank=take).float()
        if basis.ndim != 2 or basis.shape[0] == 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero contrastive rank")
        bases.append(basis.detach().contiguous())
        reports.append(
            {
                "token_id": token_id,
                "forget_context_count": int(rows.shape[0]),
                "forget_context_rank": dimension,
                "requested_contrastive_rank": FIXED_CONTRASTIVE_RANK,
                "actual_contrastive_rank": int(basis.shape[0]),
                "top_generalized_eigenvalues": [
                    float(value)
                    for value in eigenvalues[order[:take]].detach().cpu().tolist()
                ],
                "relative_eps": float(relative_eps),
                "absolute_ridge": float(ridge.detach().cpu()),
                "numerical_psd_correction": float(psd_correction.detach().cpu()),
                "total_diagonal_shift": float(diagonal_shift.detach().cpu()),
                "utility_covariance": "fixed_external_wikipedia_second_moment",
                "domain": "observed_direct_forget_context_span",
            }
        )
    return bases, reports


def total_delta_with_residual(
    stage1_delta: torch.Tensor,
    stage1_ids: Sequence[int],
    residual_delta: torch.Tensor,
    residual_ids: Sequence[int],
) -> torch.Tensor:
    total = stage1_delta.clone()
    positions = {int(token_id): index for index, token_id in enumerate(stage1_ids)}
    for index, token_id in enumerate(residual_ids):
        if int(token_id) not in positions:
            raise ValueError("residual row is absent from Stage-1 selected rows")
        total[positions[int(token_id)]] += residual_delta[index]
    return total


@torch.no_grad()
def actual_selected_delta(
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    base_rows: torch.Tensor,
) -> torch.Tensor:
    ids = torch.tensor(
        [int(x) for x in selected_ids],
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
    """Evaluate a candidate with the exact checkpoint dtype, then restore it."""
    ids = torch.tensor(
        [int(x) for x in row_ids],
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


def count_failures(state: Mapping[str, torch.Tensor], args: argparse.Namespace) -> int:
    return shared.count_failures(
        state["logit_margin"],
        state["sensitive_nll_increase"],
        required_logit_margin=args.constraint_margin,
        required_nll_increase=args.min_sensitive_nll_increase,
    )


def constraint_shortfall(
    state: Mapping[str, torch.Tensor], args: argparse.Namespace
) -> float:
    margin = torch.relu(float(args.constraint_margin) - state["logit_margin"].float())
    nll = torch.relu(
        float(args.min_sensitive_nll_increase) - state["sensitive_nll_increase"].float()
    )
    return float((margin + nll).sum().detach().cpu())


def constraint_report(
    state: Mapping[str, torch.Tensor], args: argparse.Namespace
) -> Dict[str, Any]:
    return {
        "direct_failures": count_failures(state, args),
        "constraint_shortfall_sum": constraint_shortfall(state, args),
        "minimum_logit_margin": float(state["logit_margin"].min().cpu()),
        "minimum_sensitive_nll_increase": float(
            state["sensitive_nll_increase"].min().cpu()
        ),
    }


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
            logits,
            base_logits[start : start + len(batch)],
            tids,
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


@torch.no_grad()
def evaluate_stage1_scale(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    trained_delta: torch.Tensor,
    scale: float,
    forget_cases: Sequence[core.SensitivePredictionCase],
    base_forget_logits: torch.Tensor,
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
    report = constraint_report(state, args)
    report.update(
        {
            "scale": float(scale),
            "delta_norm": float(scaled.norm().detach().cpu()),
            "direct_success": report["direct_failures"] == 0,
            "selection_inputs": "direct_constraints_only_exact_materialization",
        }
    )
    return report


def choose_stage1_report(
    reports: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], str]:
    successful = [row for row in reports if int(row["direct_failures"]) == 0]
    if successful:
        return min(successful, key=lambda row: float(row["scale"])), "complete"
    return (
        min(
            reports,
            key=lambda row: (
                int(row["direct_failures"]),
                float(row["constraint_shortfall_sum"]),
                float(row["scale"]),
            ),
        ),
        "residual_handoff",
    )


def choose_smallest_successful_scale(
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    successful = [row for row in reports if int(row["direct_failures"]) == 0]
    if not successful:
        raise RuntimeError("No candidate scale satisfies all direct constraints")
    return min(successful, key=lambda row: float(row["scale"]))


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
    llama_like: bool,
    device: torch.device,
    output_dir: Path,
) -> torch.Tensor:
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, row_bases, device=output_layer.weight.device
    )
    optimizer = torch.optim.AdamW(
        delta_module.parameters(), lr=args.stage1_lr, weight_decay=0.0
    )
    sampler = core.IndexSampler(len(forget_cases), args.stage1_batch_size, args.seed)
    hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    try:
        model.eval()
        with (output_dir / "stage1_train_log.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for step in range(1, args.stage1_steps + 1):
                indices = sampler.next()
                batch = [forget_cases[index] for index in indices]
                optimizer.zero_grad(set_to_none=True)
                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(logits, tids)
                gd = core.gd_non_sensitive_kl(logits, base_forget_logits[indices], tids)
                loss = args.ga_weight * ga + args.gd_weight * gd
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
                        "loss": float(loss.detach().cpu()),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "same_prompt_non_sensitive_gd_kl": float(gd.detach().cpu()),
                        "delta_norm": float(
                            delta_module.effective_delta().detach().norm().cpu()
                        ),
                        "benchmark_retain_examples_seen": 0,
                        "heldout_probes_seen": 0,
                    }
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
    finally:
        hook.remove()
    del optimizer
    return delta_module.effective_delta().detach().clone()


def optimize_stage2(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    stage1_ids: Sequence[int],
    stage1_delta: torch.Tensor,
    active_ids: Sequence[int],
    row_bases: Sequence[torch.Tensor],
    active_cases: Sequence[core.SensitivePredictionCase],
    active_base_logits: torch.Tensor,
    all_forget_cases: Sequence[core.SensitivePredictionCase],
    base_forget_logits: torch.Tensor,
    llama_like: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    delta_module = context.RowSpecificProjectedDelta(
        active_ids, row_bases, device=output_layer.weight.device
    )
    optimizer = torch.optim.AdamW(
        delta_module.parameters(), lr=args.stage2_lr, weight_decay=0.0
    )
    sampler = core.IndexSampler(
        len(active_cases),
        min(args.stage2_batch_size, len(active_cases)),
        args.seed + 100_003,
    )
    hook = core.register_output_delta_hook(
        output_layer, active_ids, delta_module.effective_delta
    )
    history: List[Dict[str, Any]] = []
    best_key = None
    best_delta = delta_module.effective_delta().detach().clone()
    best_report: Dict[str, Any] = {}

    def inspect(step: int, ga: Any = None, gd: Any = None) -> Dict[str, Any]:
        nonlocal best_key, best_delta, best_report
        state = shared.evaluate_shared_constraints(
            model,
            tok,
            all_forget_cases,
            base_forget_logits,
            llama_like=llama_like,
            device=device,
            batch_size=args.cache_batch_size,
        )
        current = delta_module.effective_delta().detach().clone()
        total = total_delta_with_residual(stage1_delta, stage1_ids, current, active_ids)
        row = {
            "step": int(step),
            **constraint_report(state, args),
            "residual_delta_norm": float(current.norm().cpu()),
            "total_delta_norm": float(total.norm().cpu()),
            "ga_sensitive_logprob_batch": (
                None if ga is None else float(ga.detach().cpu())
            ),
            "same_prompt_non_sensitive_gd_kl_batch": (
                None if gd is None else float(gd.detach().cpu())
            ),
            "benchmark_retain_examples_seen": 0,
            "heldout_probes_seen": 0,
        }
        history.append(row)
        key = (
            int(row["direct_failures"]),
            float(row["constraint_shortfall_sum"]),
            float(row["residual_delta_norm"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_delta = current
            best_report = dict(row)
        return row

    try:
        model.eval()
        inspect(0)
        for step in range(1, args.stage2_steps + 1):
            indices = sampler.next()
            batch = [active_cases[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(
                tok, batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, active_base_logits[indices], tids)
            loss = args.ga_weight * ga + args.gd_weight * gd
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(delta_module.parameters()), args.grad_clip
                )
            optimizer.step()

            if (
                step == 1
                or step % args.stage2_check_every == 0
                or step == args.stage2_steps
            ):
                row = inspect(step, ga, gd)
                if int(row["direct_failures"]) == 0:
                    break
    finally:
        hook.remove()
    del optimizer
    return best_delta, history, best_report


@torch.no_grad()
def evaluate_stage2_scale(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    stage1_ids: Sequence[int],
    stage1_delta: torch.Tensor,
    active_ids: Sequence[int],
    residual_delta: torch.Tensor,
    scale: float,
    forget_cases: Sequence[core.SensitivePredictionCase],
    base_forget_logits: torch.Tensor,
    llama_like: bool,
    device: torch.device,
) -> Dict[str, Any]:
    scaled = residual_delta * float(scale)
    total = total_delta_with_residual(stage1_delta, stage1_ids, scaled, active_ids)
    with temporary_materialized_output_delta(output_layer, active_ids, scaled):
        state = shared.evaluate_shared_constraints(
            model,
            tok,
            forget_cases,
            base_forget_logits,
            llama_like=llama_like,
            device=device,
            batch_size=args.cache_batch_size,
        )
    report = constraint_report(state, args)
    report.update(
        {
            "rank": FIXED_CONTRASTIVE_RANK,
            "scale": float(scale),
            "residual_delta_norm": float(scaled.norm().cpu()),
            "total_delta_norm": float(total.norm().cpu()),
            "direct_success": report["direct_failures"] == 0,
            "selection_inputs": "direct_constraints_only_exact_materialization",
        }
    )
    return report


def save_checkpoint(model: torch.nn.Module, tok: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def main() -> None:
    args = parse_args()
    scales = validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forget_records, split_manifest = load_locked(args)

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
    utility_second_moment, utility_metadata = load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=args.utility_sample_size,
        expected_hidden_size=hidden_size,
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )

    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    input_embedding_pointer = model.get_input_embeddings().weight.data_ptr()

    forget_cases = core.expand_sensitive_cases(
        forget_records,
        tok,
        dataset=args.dataset,
        llama_like=llama_like,
    )
    if not forget_cases:
        raise RuntimeError("Direct forget records produced zero token cases")
    forget_tids = core.official_target_ids(
        tok, forget_cases, llama_like=llama_like, device=device
    ).detach()
    selected_ids = sorted(
        set(int(value) for value in forget_tids.detach().cpu().tolist())
    )
    selected_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_layer.weight.device
    )
    selected_base_rows = (
        output_layer.weight.index_select(0, selected_tensor).detach().float().clone()
    )
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
    row_bases, stage1_basis_reports = build_contrastive_bases_from_second_moment(
        forget_hidden,
        forget_tids,
        utility_second_moment,
        requested_ids=selected_ids,
        rank_cap=FIXED_CONTRASTIVE_RANK,
        relative_eps=args.contrastive_eps,
    )

    torch.save(base_forget_logits, output_dir / "base_sensitive_case_logits.pt")
    core.write_json(
        output_dir / "stage1_contrastive_basis_reports.json",
        stage1_basis_reports,
    )
    core.write_json(output_dir / "utility_cache_metadata.json", utility_metadata)
    core.write_json(
        output_dir / "architecture_lock.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "dataset": args.dataset,
            "architecture_shared_across_mcf_zsre": True,
            "benchmark_retain_train_examples": 0,
            "stage1_rank": FIXED_CONTRASTIVE_RANK,
            "stage2_rank": FIXED_CONTRASTIVE_RANK,
            "stage1_objective": "sensitive_GA + same_prompt_non_sensitive_GD",
            "stage2_objective": "same objective on residual direct token cases",
            "scale_selection": (
                "smallest exact-materialized scale satisfying direct constraints"
            ),
            "utility_role": "fixed Wikipedia second moment for basis construction",
            "post_training_metrics_not_used_for_selection": [
                "official FS/GFS",
                "Spe/Spe-success",
                "official retain evaluation",
                "exact retain KL",
                "PPL",
            ],
        },
    )

    trained_stage1_delta = optimize_stage1(
        args=args,
        model=model,
        tok=tok,
        output_layer=output_layer,
        selected_ids=selected_ids,
        row_bases=row_bases,
        forget_cases=forget_cases,
        base_forget_logits=base_forget_logits,
        llama_like=llama_like,
        device=device,
        output_dir=output_dir,
    )
    torch.save(
        {"row_ids": selected_ids, "delta": trained_stage1_delta.cpu()},
        output_dir / "stage1_unscaled_delta.pt",
    )
    stage1_scale_reports = [
        evaluate_stage1_scale(
            args=args,
            model=model,
            tok=tok,
            output_layer=output_layer,
            selected_ids=selected_ids,
            trained_delta=trained_stage1_delta,
            scale=scale,
            forget_cases=forget_cases,
            base_forget_logits=base_forget_logits,
            llama_like=llama_like,
            device=device,
        )
        for scale in scales
    ]
    selected_stage1_report, stage1_scale_selection_mode = choose_stage1_report(
        stage1_scale_reports
    )
    conceptual_stage1_delta = trained_stage1_delta * float(
        selected_stage1_report["scale"]
    )
    core.materialize_output_delta(output_layer, selected_ids, conceptual_stage1_delta)
    stage1_delta = actual_selected_delta(output_layer, selected_ids, selected_base_rows)
    torch.save(
        {"row_ids": selected_ids, "delta": stage1_delta.detach().cpu()},
        output_dir / "stage1_total_delta.pt",
    )
    core.write_json(output_dir / "stage1_scale_reports.json", stage1_scale_reports)
    stage1_checkpoint = output_dir / "stage1_checkpoint"
    save_checkpoint(model, tok, stage1_checkpoint)

    stage1_state = shared.evaluate_shared_constraints(
        model,
        tok,
        forget_cases,
        base_forget_logits,
        llama_like=llama_like,
        device=device,
        batch_size=args.cache_batch_size,
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
    stage1_mode = "complete" if not active_indices else "residual_handoff"

    stage2_mode = "identity_noop"
    stage2_basis_reports: List[Dict[str, Any]] = []
    stage2_training_history: List[Dict[str, Any]] = []
    stage2_best_training_report: Dict[str, Any] | None = None
    stage2_scale_reports: List[Dict[str, Any]] = []
    selected_stage2_report: Mapping[str, Any] | None = None
    active_ids: List[int] = []
    actual_stage2_residual = output_layer.weight.new_empty((0, hidden_size)).float()

    if active_indices:
        stage2_mode = "fixed_rank2_residual_repair"
        active_cases = [forget_cases[index] for index in active_indices]
        active_hidden = forget_hidden[active_indices]
        active_tids = forget_tids[active_indices]
        active_base_logits = base_forget_logits[active_indices]
        active_ids = sorted(
            set(int(value) for value in active_tids.detach().cpu().tolist())
        )
        active_tensor = torch.tensor(
            active_ids, dtype=torch.long, device=output_layer.weight.device
        )
        active_rows_before = (
            output_layer.weight.index_select(0, active_tensor).detach().float().clone()
        )
        stage2_bases, stage2_basis_reports = build_contrastive_bases_from_second_moment(
            active_hidden,
            active_tids,
            utility_second_moment,
            requested_ids=active_ids,
            rank_cap=FIXED_CONTRASTIVE_RANK,
            relative_eps=args.contrastive_eps,
        )
        (
            residual_delta,
            stage2_training_history,
            stage2_best_training_report,
        ) = optimize_stage2(
            args=args,
            model=model,
            tok=tok,
            output_layer=output_layer,
            stage1_ids=selected_ids,
            stage1_delta=stage1_delta,
            active_ids=active_ids,
            row_bases=stage2_bases,
            active_cases=active_cases,
            active_base_logits=active_base_logits,
            all_forget_cases=forget_cases,
            base_forget_logits=base_forget_logits,
            llama_like=llama_like,
            device=device,
        )
        stage2_scale_reports = [
            evaluate_stage2_scale(
                args=args,
                model=model,
                tok=tok,
                output_layer=output_layer,
                stage1_ids=selected_ids,
                stage1_delta=stage1_delta,
                active_ids=active_ids,
                residual_delta=residual_delta,
                scale=scale,
                forget_cases=forget_cases,
                base_forget_logits=base_forget_logits,
                llama_like=llama_like,
                device=device,
            )
            for scale in scales
        ]
        core.write_json(
            output_dir / "stage2_contrastive_basis_reports.json",
            stage2_basis_reports,
        )
        core.write_json(
            output_dir / "stage2_training_history.json",
            stage2_training_history,
        )
        core.write_json(output_dir / "stage2_scale_reports.json", stage2_scale_reports)
        try:
            selected_stage2_report = choose_smallest_successful_scale(
                stage2_scale_reports
            )
        except RuntimeError:
            core.write_json(
                output_dir / "stage2_infeasible.json",
                {
                    "method": METHOD,
                    "stage1_selected": dict(selected_stage1_report),
                    "active_direct_token_cases": active_indices,
                    "stage2_best_training_report": stage2_best_training_report,
                    "stage2_scale_reports": stage2_scale_reports,
                    "selection_inputs": "direct_constraints_only",
                    "benchmark_retain_examples_seen": 0,
                    "heldout_probes_seen": 0,
                },
            )
            raise RuntimeError(
                "Fixed rank-2 Stage 2 did not satisfy all direct constraints; "
                "Stage-1 checkpoint and diagnostics were preserved"
            )
        selected_residual = residual_delta * float(selected_stage2_report["scale"])
        core.materialize_output_delta(output_layer, active_ids, selected_residual)
        active_rows_after = (
            output_layer.weight.index_select(0, active_tensor).detach().float()
        )
        actual_stage2_residual = active_rows_after - active_rows_before

    torch.save(
        {
            "row_ids": active_ids,
            "delta": actual_stage2_residual.detach().cpu(),
        },
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
    final_constraint_report = constraint_report(final_state, args)
    if int(final_constraint_report["direct_failures"]) != 0:
        raise RuntimeError("Final checkpoint failed a direct training constraint")
    if model.get_input_embeddings().weight.data_ptr() != input_embedding_pointer:
        raise RuntimeError(
            "Input embedding storage changed during sparse LM-head training"
        )
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
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_split_protocol": split_manifest.get("protocol"),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "architecture_shared_across_mcf_zsre": True,
        "transformer_trainable_parameters": 0,
        "input_embeddings_modified": False,
        "lm_head_untied": True,
        "editable_rows": "sensitive_answer_rows_only",
        "sensitive_answer_field": core.sensitive_answer_field(args.dataset),
        "replacement_or_reference_target_used": False,
        "benchmark_retain_train_examples": 0,
        "benchmark_retain_answer_labels_seen": False,
        "official_retain_eval_seen_during_training_or_selection": 0,
        "heldout_probes_seen_during_training_or_selection": 0,
        "ppl_text_seen_during_training_or_selection": 0,
        "wikipedia_utility_examples_are_benchmark_retain": False,
        "utility_cache": str(Path(args.utility_cache).resolve()),
        "utility_cache_metadata": utility_metadata,
        "utility_role": "contrastive_basis_only",
        "utility_used_as_training_loss": False,
        "contrastive_generalized_eigenproblem": True,
        "contrastive_domain": "observed_direct_forget_context_span",
        "stage1_contrastive_rank": FIXED_CONTRASTIVE_RANK,
        "stage2_contrastive_rank": FIXED_CONTRASTIVE_RANK,
        "contrastive_eps": float(args.contrastive_eps),
        "stage1_objective": "sensitive_GA + same_prompt_non_sensitive_GD",
        "stage2_objective": (
            "same sensitive_GA + same_prompt_non_sensitive_GD on failed "
            "direct token cases only"
        ),
        "ga_weight": float(args.ga_weight),
        "gd_weight": float(args.gd_weight),
        "retain_action_loss_present": False,
        "retain_action_budget_present": False,
        "delta_norm_guard_present": False,
        "kl_guard_present": False,
        "candidate_rank_sweep_present": False,
        "constraint_margin": float(args.constraint_margin),
        "min_sensitive_nll_increase": float(args.min_sensitive_nll_increase),
        "candidate_scales": scales,
        "scale_selection_rule": (
            "smallest exact-materialized candidate scale with zero direct "
            "failures; Stage-1 handoff uses failures then direct-constraint "
            "shortfall"
        ),
        "scale_selection_uses_only_direct_constraints": True,
        "stage1_row_basis_reports": stage1_basis_reports,
        "stage1_scale_reports": stage1_scale_reports,
        "stage1_selected": dict(selected_stage1_report),
        "stage1_scale_selection_mode": stage1_scale_selection_mode,
        "stage1_mode": stage1_mode,
        "stage1_direct_failures_after_materialization": len(active_indices),
        "stage1_failed_direct_token_indices": active_indices,
        "stage2_mode": stage2_mode,
        "stage2_active_sensitive_row_ids": active_ids,
        "stage2_row_basis_reports": stage2_basis_reports,
        "stage2_training_history": stage2_training_history,
        "stage2_best_training_report": stage2_best_training_report,
        "stage2_scale_reports": stage2_scale_reports,
        "stage2_selected": (
            None if selected_stage2_report is None else dict(selected_stage2_report)
        ),
        "final_direct_constraints": final_constraint_report,
        "final_total_delta_norm": float(final_delta.norm().detach().cpu()),
        "final_same_prompt_non_sensitive_gd_kl": final_forget_gd,
        "final_wikipedia_logit_drift_mse_posthoc": final_utility_drift,
        "post_training_only_metrics": [
            "official FS/GFS",
            "Spe/Spe-success and Spe-margin",
            "official retain metrics",
            "exact official-retain sparse-row KL",
            "PPL",
        ],
        "stage1_checkpoint": str(stage1_checkpoint.resolve()),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_delta": str((output_dir / "final_total_delta.pt").resolve()),
    }
    core.write_json(output_dir / "config_used.json", config)

    print("Minimal two-stage SURE-LM complete:", final_checkpoint)
    print("dataset:", args.dataset)
    print("benchmark retain examples used for training: 0")
    print("fixed Stage-1/Stage-2 rank:", FIXED_CONTRASTIVE_RANK)
    print("Stage-1 selected scale:", selected_stage1_report["scale"])
    print("Stage-1 residual direct token cases:", len(active_indices))
    print("Stage-2 mode:", stage2_mode)
    if selected_stage2_report is not None:
        print("Stage-2 selected scale:", selected_stage2_report["scale"])
    print("Final direct failures:", final_constraint_report["direct_failures"])
    print("Final total delta norm:", config["final_total_delta_norm"])
    print("Post-training evaluation has not run yet")


if __name__ == "__main__":
    main()
