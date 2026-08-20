#!/usr/bin/env python3
"""MCF-specific target-aware SURE with joint true-GA/new-GD training.

This is an explicitly benchmark-aware ablation.  It uses the sampled MCF
``target_true`` and ``target_new`` answers *and* the official paraphrase prompts
during optimization and checkpoint selection.  It must therefore never be
reported as the benchmark-neutral SURE result.

The frozen transformer supplies fixed predictor states.  Stage 1 learns sparse
LM-head row edits with four bounded objectives:

* increase NLL(target_true) (bounded gradient ascent / suppression),
* decrease NLL(target_new) (bounded gradient descent / installation),
* make NLL(target_true) - NLL(target_new) positive on direct and paraphrase
  prompts, and
* preserve an external Wikipedia distribution.

Stage 2 touches only rows implicated by remaining direct/paraphrase failures
and solves a minimum-utility residual subject to exact sequence-level pairwise
constraints.  Candidate solver margins are deliberately much larger than the
paper's strict ``> 0`` rule.  Every candidate is then physically materialized
in checkpoint dtype and scored by the official MCF routine.  A checkpoint is
emitted only when the materialized model has FS=100 and GFS=100, passes the
requested positive margin on every prompt instance, and satisfies all utility
guards.
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
import torch.nn.functional as F
from scipy.optimize import minimize

import build_sure_minimal_split as split_builder
import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import mcf_zero_unlearn_official_eval as mcf_official
import sure_canonical_core as core
import sure_mcf_direct_fs_repair as exact
import sure_minimal_two_stage as learner


METHOD = "SURE-LM-MCF-target-aware-true-GA-new-GD-v7"
PROTOCOL = "sure_mcf_target_aware_true_ga_new_gd_direct_paraphrase_v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--utility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--utility-sample-size", type=int, default=100_000)
    parser.add_argument("--utility-prompt-count", type=int, default=100_000)
    parser.add_argument("--utility-token-topk-per-row", type=int, default=128)
    parser.add_argument("--utility-uniform-prompt-count", type=int, default=1_024)
    parser.add_argument("--utility-pool-seed", type=int, default=1)
    parser.add_argument("--utility-train-batch-size", type=int, default=128)
    parser.add_argument("--utility-eval-batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=8)

    parser.add_argument("--stage1-rank", type=int, default=4)
    parser.add_argument("--stage1-steps", type=int, default=600)
    parser.add_argument("--stage1-lr", type=float, default=5e-3)
    parser.add_argument("--stage1-pairwise-target", type=float, default=1.0)
    parser.add_argument("--stage1-true-nll-increase", type=float, default=2.0)
    parser.add_argument("--stage1-new-nll-decrease", type=float, default=1.0)
    parser.add_argument("--stage1-pairwise-weight", type=float, default=100.0)
    parser.add_argument("--stage1-true-ga-weight", type=float, default=10.0)
    parser.add_argument("--stage1-new-gd-weight", type=float, default=10.0)
    parser.add_argument("--stage1-utility-kl-weight", type=float, default=1.0)
    parser.add_argument("--stage1-l2-weight", type=float, default=1e-4)
    parser.add_argument(
        "--stage1-candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.125,0",
    )

    parser.add_argument("--required-pairwise-margin", type=float, default=0.01)
    parser.add_argument(
        "--stage2-solver-margins",
        default="0.5,1.0,2.0",
        help=(
            "Absolute continuous direct/paraphrase separation targets tried in "
            "order. These are BF16 safety targets, not paper metric changes."
        ),
    )
    parser.add_argument("--stage2-rank-ladder", default="2,4,8")
    parser.add_argument("--stage2-maxiter", type=int, default=500)
    parser.add_argument("--stage2-ftol", type=float, default=1e-9)
    parser.add_argument("--stage2-constraint-tolerance", type=float, default=1e-5)
    parser.add_argument("--stage2-residual-l2-weight", type=float, default=1e-4)
    parser.add_argument("--constraint-context-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-eps", type=float, default=1e-3)

    parser.add_argument("--utility-kl-mean-budget", type=float, default=0.01)
    parser.add_argument("--utility-kl-p95-budget", type=float, default=0.05)
    parser.add_argument("--utility-kl-max-budget", type=float, default=0.5)
    parser.add_argument("--max-total-delta-norm", type=float, default=1.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def parse_unique_numbers(text: str, cast: Any, *, positive: bool) -> Tuple[Any, ...]:
    values: List[Any] = []
    for raw in str(text).split(","):
        value = cast(raw.strip())
        if not math.isfinite(float(value)):
            raise ValueError("numeric ladder values must be finite")
        if positive and value <= 0:
            raise ValueError("numeric ladder values must be positive")
        if not positive and value < 0:
            raise ValueError("candidate scales must be non-negative")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("numeric ladder must not be empty")
    return tuple(values)


def validate_args(
    args: argparse.Namespace,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    positive_names = (
        "forget_num",
        "utility_sample_size",
        "utility_prompt_count",
        "utility_token_topk_per_row",
        "utility_train_batch_size",
        "utility_eval_batch_size",
        "cache_batch_size",
        "stage1_rank",
        "stage1_steps",
        "stage1_lr",
        "stage2_maxiter",
        "stage2_ftol",
        "contrastive_eps",
        "required_pairwise_margin",
        "max_total_delta_norm",
    )
    for name in positive_names:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    nonnegative_names = (
        "utility_uniform_prompt_count",
        "stage1_pairwise_target",
        "stage1_true_nll_increase",
        "stage1_new_nll_decrease",
        "stage1_pairwise_weight",
        "stage1_true_ga_weight",
        "stage1_new_gd_weight",
        "stage1_utility_kl_weight",
        "stage1_l2_weight",
        "stage2_constraint_tolerance",
        "stage2_residual_l2_weight",
        "constraint_context_weight",
        "utility_kl_mean_budget",
        "utility_kl_p95_budget",
        "utility_kl_max_budget",
    )
    for name in nonnegative_names:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    scales = parse_unique_numbers(args.stage1_candidate_scales, float, positive=False)
    if 0.0 not in scales or 1.0 not in scales:
        raise ValueError("Stage-1 candidate scales must contain 0 and 1")
    margins = parse_unique_numbers(args.stage2_solver_margins, float, positive=True)
    if any(value <= float(args.required_pairwise_margin) for value in margins):
        raise ValueError("every Stage-2 solver margin must exceed the reported margin")
    if float(args.stage1_pairwise_target) < float(args.required_pairwise_margin):
        raise ValueError("Stage-1 pairwise target must cover the reported margin")
    if not (
        float(args.utility_kl_mean_budget)
        <= float(args.utility_kl_p95_budget)
        <= float(args.utility_kl_max_budget)
    ):
        raise ValueError("utility KL budgets must satisfy mean <= p95 <= max")
    ranks = parse_unique_numbers(args.stage2_rank_ladder, int, positive=True)
    return scales, margins, ranks


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_target_aware_records(
    mcf_path: Path,
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source_bytes = mcf_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "mcf":
        raise RuntimeError("target-aware GA/GD requires an MCF manifest")
    if manifest.get("source_sha256") != sha256_bytes(source_bytes):
        raise RuntimeError("MCF source does not match the split manifest")
    raw = json.loads(source_bytes)
    case_ids = manifest.get("sampling", {}).get("forget_case_ids", [])
    if not isinstance(case_ids, list) or not case_ids:
        raise RuntimeError("split manifest lacks forget_case_ids")

    originals: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []
    for record_position, raw_case_id in enumerate(case_ids):
        case_id = int(raw_case_id)
        if case_id < 0 or case_id >= len(raw):
            raise RuntimeError(f"MCF case id is out of range: {case_id}")
        original = mcf_official.normalize_record(raw[case_id])
        rewrite = split_builder.normalize_mcf_rewrite(original)
        target_true = str(rewrite.get("target_true", {}).get("str", "")).strip()
        target_new = str(rewrite.get("target_new", {}).get("str", "")).strip()
        if not target_true or not target_new:
            raise RuntimeError(f"MCF case {case_id} lacks target_true/target_new")
        subject = str(rewrite["subject"])
        direct_prompt = str(rewrite["prompt"]).format(subject)
        paraphrases = original.get("paraphrase_prompts", [])
        if not isinstance(paraphrases, list) or not paraphrases:
            raise RuntimeError(
                f"MCF case {case_id} has no official paraphrases; GFS cannot be trained"
            )
        originals.append(original)
        prompt_specs = [("direct", 0, direct_prompt)] + [
            ("paraphrase", index, str(prompt))
            for index, prompt in enumerate(paraphrases)
        ]
        for kind, prompt_index, prompt in prompt_specs:
            prompts.append(
                {
                    "case_id": case_id,
                    "source_record_position": record_position,
                    "prompt_kind": kind,
                    "prompt_index": int(prompt_index),
                    "prompt_text": prompt,
                    "requested_rewrite": {
                        "prompt": "{}",
                        "subject": prompt,
                        "target_sensitive": {"str": target_true},
                        "target_reference": {"str": target_new},
                    },
                }
            )
    return originals, prompts, manifest


def prompt_kind_masks(
    prompt_records: Sequence[Mapping[str, Any]], *, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        kind: torch.tensor(
            [record["prompt_kind"] == kind for record in prompt_records],
            device=device,
            dtype=torch.bool,
        )
        for kind in ("direct", "paraphrase")
    }


def balanced_direct_paraphrase_mean(
    values: torch.Tensor, masks: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    means: List[torch.Tensor] = []
    for kind in ("direct", "paraphrase"):
        mask = masks[kind].to(device=values.device)
        if not bool(mask.any()):
            raise ValueError(f"no {kind} prompt instances")
        means.append(values[mask].mean())
    return torch.stack(means).mean()


def grouped_pairwise_report(
    separation: torch.Tensor,
    prompt_records: Sequence[Mapping[str, Any]],
    *,
    required_margin: float,
) -> Dict[str, Any]:
    values = separation.detach().float().cpu()
    if values.shape != (len(prompt_records),):
        raise ValueError("pairwise values do not align with prompt records")
    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {}
    margin_failure_positions: List[int] = []
    for position, (value, record) in enumerate(zip(values.tolist(), prompt_records)):
        success = float(value) > 0.0
        margin_safe = float(value) >= float(required_margin)
        if not margin_safe:
            margin_failure_positions.append(position)
        rows.append(
            {
                "prompt_position": position,
                "case_id": int(record["case_id"]),
                "source_record_position": int(record["source_record_position"]),
                "prompt_kind": str(record["prompt_kind"]),
                "prompt_index": int(record["prompt_index"]),
                "separation": float(value),
                "success": success,
                "margin_safe": margin_safe,
            }
        )
    for kind, metric in (("direct", "FS"), ("paraphrase", "GFS")):
        subset = [row for row in rows if row["prompt_kind"] == kind]
        failures = sum(not row["success"] for row in subset)
        margin_failures = sum(not row["margin_safe"] for row in subset)
        report[metric] = 100.0 * (len(subset) - failures) / len(subset)
        report[f"{kind}_prompt_count"] = len(subset)
        report[f"{kind}_failures"] = failures
        report[f"{kind}_margin_failures"] = margin_failures
        report[f"minimum_{kind}_separation"] = min(
            float(row["separation"]) for row in subset
        )
        report[f"mean_{kind}_separation"] = sum(
            float(row["separation"]) for row in subset
        ) / len(subset)
    report["required_pairwise_margin"] = float(required_margin)
    report["pairwise_margin_failure_positions"] = margin_failure_positions
    report["minimum_overall_separation"] = float(values.min().item())
    report["prompt_instances"] = rows
    return report


@torch.no_grad()
def official_materialized_report(
    model: torch.nn.Module,
    tok: Any,
    originals: Sequence[Mapping[str, Any]],
    prompt_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    required_margin: float,
) -> Dict[str, Any]:
    keyed: Dict[Tuple[int, str, int], float] = {}
    for record_position, original in enumerate(originals):
        rewrite = split_builder.normalize_mcf_rewrite(original)
        direct = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        paraphrases = [str(value) for value in original.get("paraphrase_prompts", [])]
        prefixes = [direct, *paraphrases]
        scores = mcf_official.official_test_batch_prediction(
            model,
            tok,
            prefixes,
            str(rewrite["target_new"]["str"]),
            str(rewrite["target_true"]["str"]),
            device,
            llama_like=llama_like,
        )
        for local_index, score in enumerate(scores):
            kind = "direct" if local_index == 0 else "paraphrase"
            prompt_index = 0 if local_index == 0 else local_index - 1
            keyed[(record_position, kind, prompt_index)] = float(
                score["target_true"] - score["target_new"]
            )
    separations = []
    for record in prompt_records:
        key = (
            int(record["source_record_position"]),
            str(record["prompt_kind"]),
            int(record["prompt_index"]),
        )
        if key not in keyed:
            raise RuntimeError(f"official scorer omitted prompt {key}")
        separations.append(keyed[key])
    report = grouped_pairwise_report(
        torch.tensor(separations, dtype=torch.float32),
        prompt_records,
        required_margin=required_margin,
    )
    report["scorer"] = "mcf_zero_unlearn_official_eval.official_test_batch_prediction"
    report["checkpoint_dtype_forward"] = True
    return report


@torch.no_grad()
def build_joint_bases(
    true_hidden: torch.Tensor,
    true_ids: torch.Tensor,
    reference_hidden: torch.Tensor,
    reference_ids: torch.Tensor,
    utility_second_moment: torch.Tensor,
    *,
    requested_ids: Sequence[int],
    rank_cap: int,
    relative_eps: float,
    constraint_context_weight: float,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    if rank_cap <= 0:
        raise ValueError("rank cap must be positive")
    device = true_hidden.device
    cholesky, _ = learner.regularized_utility_cholesky(
        utility_second_moment, relative_eps=relative_eps, device=device
    )
    all_hidden = torch.cat((true_hidden.float(), reference_hidden.float()), dim=0)
    whitened_all = torch.linalg.solve_triangular(
        cholesky, all_hidden.transpose(0, 1), upper=False
    ).transpose(0, 1) / math.sqrt(float(all_hidden.shape[0]))
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for token_id in [int(value) for value in requested_ids]:
        true_rows = true_hidden[true_ids.eq(token_id)].float()
        reference_rows = reference_hidden[reference_ids.eq(token_id)].float()
        relevant = torch.cat((true_rows, reference_rows), dim=0)
        if relevant.numel() == 0:
            raise RuntimeError(f"edited row {token_id} has no target contexts")
        whitened_relevant = torch.linalg.solve_triangular(
            cholesky, relevant.transpose(0, 1), upper=False
        ).transpose(0, 1) / math.sqrt(float(relevant.shape[0]))
        components = [whitened_relevant]
        if constraint_context_weight > 0:
            components.append(math.sqrt(constraint_context_weight) * whitened_all)
        whitened = torch.cat(components, dim=0)
        _, singular_values, right = torch.linalg.svd(whitened, full_matrices=False)
        tolerance = (
            max(whitened.shape)
            * torch.finfo(torch.float32).eps
            * singular_values.max().clamp_min(1.0)
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
        take = min(rank_cap, numerical_rank)
        if take <= 0:
            raise RuntimeError(f"edited row {token_id} has zero numerical rank")
        raw = torch.linalg.solve_triangular(
            cholesky.transpose(0, 1), right[:take].transpose(0, 1), upper=True
        ).transpose(0, 1)
        basis = core.orthonormal_row_basis(raw, max_rank=take).float()
        bases.append(basis.detach().contiguous())
        reports.append(
            {
                "token_id": token_id,
                "true_context_count": int(true_rows.shape[0]),
                "reference_context_count": int(reference_rows.shape[0]),
                "all_constraint_context_count": int(all_hidden.shape[0]),
                "requested_rank": int(rank_cap),
                "actual_rank": int(basis.shape[0]),
                "numerical_rank": numerical_rank,
                "constraint_context_weight": float(constraint_context_weight),
                "basis_protocol": "utility_whitened_joint_true_new_plus_all_contexts",
            }
        )
    return bases, reports


def stage1_losses(
    true_cache: Mapping[str, Any],
    reference_cache: Mapping[str, Any],
    delta: torch.Tensor,
    base_true_nll: torch.Tensor,
    base_reference_nll: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    *,
    pairwise_target: float,
    true_nll_increase_target: float,
    new_nll_decrease_target: float,
) -> Dict[str, torch.Tensor]:
    true_nll = exact.exact_sequence_record_nll(true_cache, delta)
    reference_nll = exact.exact_sequence_record_nll(reference_cache, delta)
    true_increase = true_nll - base_true_nll
    new_decrease = base_reference_nll - reference_nll
    separation = true_nll - reference_nll
    pairwise = balanced_direct_paraphrase_mean(
        F.relu(float(pairwise_target) - separation).square(), masks
    )
    true_ga = balanced_direct_paraphrase_mean(
        F.relu(float(true_nll_increase_target) - true_increase).square(), masks
    )
    new_gd = balanced_direct_paraphrase_mean(
        F.relu(float(new_nll_decrease_target) - new_decrease).square(), masks
    )
    return {
        "pairwise": pairwise,
        "true_ga": true_ga,
        "new_gd": new_gd,
        "true_nll": true_nll,
        "reference_nll": reference_nll,
        "true_nll_increase": true_increase,
        "new_nll_decrease": new_decrease,
        "separation": separation,
    }


def optimize_stage1(
    args: argparse.Namespace,
    selected_ids: Sequence[int],
    row_bases: Sequence[torch.Tensor],
    true_cache: Mapping[str, Any],
    reference_cache: Mapping[str, Any],
    base_true_nll: torch.Tensor,
    base_reference_nll: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    output_dir: Path,
    *,
    device: torch.device,
) -> torch.Tensor:
    module = learner.context.RowSpecificProjectedDelta(
        selected_ids, row_bases, device=device
    )
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=args.stage1_lr, weight_decay=0
    )
    sampler = core.IndexSampler(
        int(utility_hidden.shape[0]),
        min(int(args.utility_train_batch_size), int(utility_hidden.shape[0])),
        int(args.seed) + 7919,
    )
    log_path = output_dir / "stage1_training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for step in range(1, int(args.stage1_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = module.effective_delta()
            components = stage1_losses(
                true_cache,
                reference_cache,
                delta,
                base_true_nll,
                base_reference_nll,
                masks,
                pairwise_target=args.stage1_pairwise_target,
                true_nll_increase_target=args.stage1_true_nll_increase,
                new_nll_decrease_target=args.stage1_new_nll_decrease,
            )
            indices = sampler.next()
            utility_kl = learner.exact_sparse_utility_kl(
                delta,
                utility_hidden[indices],
                utility_probabilities[indices],
            ).mean()
            loss = (
                float(args.stage1_pairwise_weight) * components["pairwise"]
                + float(args.stage1_true_ga_weight) * components["true_ga"]
                + float(args.stage1_new_gd_weight) * components["new_gd"]
                + float(args.stage1_utility_kl_weight) * utility_kl
                + float(args.stage1_l2_weight) * delta.square().sum()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite target-aware Stage-1 loss at {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == int(args.stage1_steps):
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pairwise_hinge": float(components["pairwise"].detach().cpu()),
                    "bounded_target_true_GA": float(
                        components["true_ga"].detach().cpu()
                    ),
                    "bounded_target_new_GD": float(components["new_gd"].detach().cpu()),
                    "wikipedia_exact_kl": float(utility_kl.detach().cpu()),
                    "minimum_pairwise_separation": float(
                        components["separation"].min().detach().cpu()
                    ),
                    "minimum_true_nll_increase": float(
                        components["true_nll_increase"].min().detach().cpu()
                    ),
                    "minimum_new_nll_decrease": float(
                        components["new_nll_decrease"].min().detach().cpu()
                    ),
                    "delta_norm": float(delta.detach().norm().cpu()),
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
    return module.effective_delta().detach().clone()


def add_utility_report(
    report: Dict[str, Any],
    delta: torch.Tensor,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> None:
    values = learner.utility_kl_report(
        delta,
        utility_hidden,
        utility_probabilities,
        device=device,
        batch_size=args.utility_eval_batch_size,
    )
    report.update(values)
    report["total_delta_norm"] = float(delta.norm().detach().cpu())
    checks = {
        "mean": values["utility_kl_mean"] <= float(args.utility_kl_mean_budget),
        "p95": values["utility_kl_p95"] <= float(args.utility_kl_p95_budget),
        "max": values["utility_kl_max"] <= float(args.utility_kl_max_budget),
        "norm": report["total_delta_norm"] <= float(args.max_total_delta_norm),
    }
    report["utility_guard_checks"] = checks
    report["utility_safe"] = bool(all(checks.values()))


def choose_stage1_delta(
    args: argparse.Namespace,
    trained_delta: torch.Tensor,
    scales: Sequence[float],
    true_cache: Mapping[str, Any],
    reference_cache: Mapping[str, Any],
    prompt_records: Sequence[Mapping[str, Any]],
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    reports: List[Dict[str, Any]] = []
    for scale in scales:
        delta = trained_delta * float(scale)
        separation = exact.exact_pairwise_separation(true_cache, reference_cache, delta)
        report = {
            "scale": float(scale),
            **grouped_pairwise_report(
                separation,
                prompt_records,
                required_margin=args.required_pairwise_margin,
            ),
        }
        report.pop("prompt_instances", None)
        add_utility_report(
            report,
            delta,
            utility_hidden,
            utility_probabilities,
            args,
            device=device,
        )
        reports.append(report)
    safe = [report for report in reports if report["utility_safe"]]
    if not safe:
        raise RuntimeError("no Stage-1 scale passed the Wikipedia/norm guards")
    complete = [
        report
        for report in safe
        if report["direct_margin_failures"] == 0
        and report["paraphrase_margin_failures"] == 0
    ]
    if complete:
        selected = min(
            complete,
            key=lambda row: (
                float(row["utility_kl_mean"]),
                float(row["total_delta_norm"]),
            ),
        )
        mode = "joint_stage1_complete"
    else:
        selected = min(
            safe,
            key=lambda row: (
                int(row["direct_margin_failures"])
                + int(row["paraphrase_margin_failures"]),
                -float(row["minimum_overall_separation"]),
                float(row["utility_kl_mean"]),
            ),
        )
        mode = "joint_stage1_residual_handoff"
    selected = dict(selected)
    selected["selection_mode"] = mode
    return trained_delta * float(selected["scale"]), selected, reports


def solve_residual(
    args: argparse.Namespace,
    *,
    rank: int,
    solver_target: float,
    row_bases: Sequence[torch.Tensor],
    active_ids: Sequence[int],
    selected_ids: Sequence[int],
    stage1_delta: torch.Tensor,
    true_cache: Mapping[str, Any],
    reference_cache: Mapping[str, Any],
    prompt_records: Sequence[Mapping[str, Any]],
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    device = stage1_delta.device
    bases = [basis.to(device=device, dtype=torch.float32) for basis in row_bases]
    coefficient_count = sum(int(basis.shape[0]) for basis in bases)

    def residual(coefficients: torch.Tensor) -> torch.Tensor:
        return learner.coefficients_to_residual(coefficients, bases)

    def total(coefficients: torch.Tensor) -> torch.Tensor:
        return learner.total_delta_with_residual(
            stage1_delta, selected_ids, residual(coefficients), active_ids
        )

    def utility_values(coefficients: torch.Tensor) -> torch.Tensor:
        return learner.exact_sparse_utility_kl(
            total(coefficients), utility_hidden, utility_probabilities
        )

    def objective(coefficients: torch.Tensor) -> torch.Tensor:
        return (
            utility_values(coefficients).mean()
            + float(args.stage2_residual_l2_weight)
            * residual(coefficients).square().sum()
        )

    def behavioral_slacks(coefficients: torch.Tensor) -> torch.Tensor:
        separation = exact.exact_pairwise_separation(
            true_cache, reference_cache, total(coefficients)
        )
        return separation - float(solver_target)

    def utility_slacks(coefficients: torch.Tensor) -> torch.Tensor:
        values = utility_values(coefficients)
        combined = total(coefficients)
        return torch.stack(
            (
                values.new_tensor(float(args.utility_kl_mean_budget)) - values.mean(),
                values.new_tensor(float(args.utility_kl_p95_budget))
                - torch.quantile(values, 0.95),
                values.new_tensor(float(args.utility_kl_max_budget)) - values.max(),
                values.new_tensor(float(args.max_total_delta_norm)) - combined.norm(),
            )
        )

    scalar = learner._TorchScalarAdapter(objective, device=device)
    behavioral = learner._TorchVectorAdapter(behavioral_slacks, device=device)
    utility = learner._TorchVectorAdapter(utility_slacks, device=device)
    zero = np.zeros(coefficient_count, dtype=np.float64)
    initial_slacks = behavioral.value(zero)
    initial_jacobian = behavioral.jacobian(zero)
    violated = initial_slacks < 0
    starts = [zero]
    if bool(np.any(violated)):
        rhs = -initial_slacks[violated] + 0.05
        directed, *_ = np.linalg.lstsq(initial_jacobian[violated], rhs, rcond=None)
        if np.isfinite(directed).all() and not np.allclose(directed, zero):
            starts.append(np.asarray(directed, dtype=np.float64))

    history: List[Dict[str, Any]] = []
    observed: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    tolerance = float(args.stage2_constraint_tolerance)

    def inspect(
        values: np.ndarray, restart: int, iteration: int, phase: str
    ) -> Dict[str, Any]:
        coefficients = torch.tensor(values, device=device, dtype=torch.float32)
        combined = total(coefficients).detach()
        separation = exact.exact_pairwise_separation(
            true_cache, reference_cache, combined
        ).detach()
        behavior = behavioral_slacks(coefficients).detach()
        utility_now = utility_values(coefficients).detach().double().cpu()
        utility_constraints = utility_slacks(coefficients).detach()
        row = {
            "phase": phase,
            "restart": restart,
            "iteration": iteration,
            "rank": rank,
            "solver_target": float(solver_target),
            **grouped_pairwise_report(
                separation,
                prompt_records,
                required_margin=args.required_pairwise_margin,
            ),
            "minimum_behavioral_slack": float(behavior.min().cpu()),
            "minimum_utility_slack": float(utility_constraints.min().cpu()),
            "utility_kl_mean": float(utility_now.mean()),
            "utility_kl_p95": float(torch.quantile(utility_now, 0.95)),
            "utility_kl_max": float(utility_now.max()),
            "residual_delta_norm": float(residual(coefficients).detach().norm().cpu()),
            "total_delta_norm": float(combined.norm().cpu()),
            "objective": float(objective(coefficients).detach().cpu()),
            "continuous_feasible": bool(
                float(behavior.min().cpu()) >= -tolerance
                and float(utility_constraints.min().cpu()) >= -tolerance
            ),
        }
        row.pop("prompt_instances", None)
        history.append(row)
        observed.append((coefficients.detach().cpu(), row))
        return row

    attempts: List[Dict[str, Any]] = []
    for restart, start in enumerate(starts):
        iteration = 0
        inspect(start, restart, iteration, "initial")

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
                {"type": "ineq", "fun": behavioral.value, "jac": behavioral.jacobian},
                {"type": "ineq", "fun": utility.value, "jac": utility.jacobian},
            ),
            callback=callback,
            options={
                "maxiter": int(args.stage2_maxiter),
                "ftol": float(args.stage2_ftol),
                "disp": False,
            },
        )
        final = inspect(result.x, restart, iteration + 1, "final")
        attempts.append(
            {
                "restart": restart,
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
                float(item[1]["total_delta_norm"]),
            ),
        )
        selection_mode = "minimum_utility_exact_FS_GFS_feasible"
    else:
        coefficients, report = min(
            observed,
            key=lambda item: (
                int(item[1]["direct_margin_failures"])
                + int(item[1]["paraphrase_margin_failures"]),
                -float(item[1]["minimum_overall_separation"]),
                float(item[1]["objective"]),
            ),
        )
        selection_mode = "best_infeasible_diagnostic"
    chosen = coefficients.to(device=device, dtype=torch.float32)
    residual_delta = residual(chosen).detach()
    solver_report = dict(report)
    solver_report.update(
        {
            "selection_mode": selection_mode,
            "coefficient_count": coefficient_count,
            "continuous_feasible": bool(report["continuous_feasible"]),
            "solver_attempts": attempts,
            "hard_constraints_tradeable": False,
        }
    )
    return residual_delta, history, solver_report


def main() -> None:
    args = parse_args()
    stage1_scales, solver_margins, rank_ladder = validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    originals, prompt_records, manifest = load_target_aware_records(
        Path(args.mcf_path).resolve(), Path(args.split_manifest).resolve()
    )
    if len(originals) != int(args.forget_num):
        raise RuntimeError("manifest forget count differs from --forget-num")

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
    original_output = model.get_output_embeddings()
    if original_output is None:
        raise RuntimeError("model has no output head")
    hidden_size = int(original_output.weight.shape[1])
    (
        utility_second_moment,
        utility_hidden,
        utility_logsumexp,
        utility_metadata,
    ) = learner.load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=args.utility_sample_size,
        expected_prompt_count=args.utility_prompt_count,
        expected_hidden_size=hidden_size,
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )
    actual_documents = int(utility_metadata["actual_document_sample_size"])
    actual_prompts = int(utility_metadata["actual_utility_prompt_count"])
    if actual_documents < int(args.utility_sample_size):
        print(
            "WARNING: Wikipedia corpus contains only "
            f"{actual_documents} sampled documents versus "
            f"{args.utility_sample_size} requested; this is a pilot utility guard"
        )
    if actual_prompts < int(args.utility_prompt_count):
        print(
            "WARNING: Wikipedia cache contains only "
            f"{actual_prompts} predictor states versus "
            f"{args.utility_prompt_count} requested"
        )
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    true_cases = core.expand_sensitive_cases(
        prompt_records,
        tok,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    reference_cases = core.expand_sensitive_cases(
        prompt_records,
        tok,
        sensitive_field="target_reference",
        llama_like=llama_like,
    )
    true_ids = core.official_target_ids(
        tok, true_cases, llama_like=llama_like, device=device
    ).detach()
    reference_ids = core.official_target_ids(
        tok, reference_cases, llama_like=llama_like, device=device
    ).detach()
    selected_ids = sorted(
        set(int(value) for value in true_ids.cpu().tolist())
        | set(int(value) for value in reference_ids.cpu().tolist())
    )
    selected_tensor = torch.tensor(
        selected_ids, device=output_layer.weight.device, dtype=torch.long
    )
    base_rows = output_layer.weight.index_select(0, selected_tensor).detach().float()

    utility_probabilities = learner.selected_base_probabilities(
        output_layer,
        selected_ids,
        utility_hidden,
        utility_logsumexp,
        device=device,
        batch_size=args.utility_eval_batch_size,
    )
    (
        train_indices,
        guard_indices,
        utility_pool_report,
    ) = learner.build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=utility_probabilities,
        selected_ids=selected_ids,
        topk_per_row=args.utility_token_topk_per_row,
        uniform_prompt_count=args.utility_uniform_prompt_count,
        split_seed=args.utility_pool_seed,
    )
    utility_train_hidden = (
        utility_hidden.index_select(0, train_indices).contiguous().to(device)
    )
    utility_train_probabilities = (
        utility_probabilities.index_select(0, train_indices).contiguous().to(device)
    )
    utility_guard_hidden = utility_hidden.index_select(0, guard_indices).contiguous()
    utility_guard_probabilities = utility_probabilities.index_select(
        0, guard_indices
    ).contiguous()
    core.write_json(output_dir / "utility_pool_report.json", utility_pool_report)

    base_true_logits = learner.cache_logits_preserving_dtype(
        model, tok, true_cases, device, args.cache_batch_size
    )
    base_reference_logits = learner.cache_logits_preserving_dtype(
        model, tok, reference_cases, device, args.cache_batch_size
    )
    true_hidden = core.forward_last_hidden(
        model, tok, true_cases, device, args.cache_batch_size
    ).float()
    reference_hidden = core.forward_last_hidden(
        model, tok, reference_cases, device, args.cache_batch_size
    ).float()
    true_positions = exact.record_positions(true_cases, device=device)
    reference_positions = exact.record_positions(reference_cases, device=device)
    prompt_count = len(prompt_records)
    true_cache = exact.build_sequence_cache(
        base_true_logits,
        true_hidden,
        true_ids,
        true_positions,
        selected_ids,
        record_count=prompt_count,
        device=device,
    )
    reference_cache = exact.build_sequence_cache(
        base_reference_logits,
        reference_hidden,
        reference_ids,
        reference_positions,
        selected_ids,
        record_count=prompt_count,
        device=device,
    )
    zero = torch.zeros(
        (len(selected_ids), hidden_size), device=device, dtype=torch.float32
    )
    base_true_nll = exact.exact_sequence_record_nll(true_cache, zero).detach()
    base_reference_nll = exact.exact_sequence_record_nll(reference_cache, zero).detach()
    masks = prompt_kind_masks(prompt_records, device=device)
    base_report = official_materialized_report(
        model,
        tok,
        originals,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=args.required_pairwise_margin,
    )
    core.write_json(output_dir / "base_FS_GFS_report.json", base_report)

    stage1_bases, stage1_basis_report = build_joint_bases(
        true_hidden,
        true_ids,
        reference_hidden,
        reference_ids,
        utility_second_moment,
        requested_ids=selected_ids,
        rank_cap=args.stage1_rank,
        relative_eps=args.contrastive_eps,
        constraint_context_weight=args.constraint_context_weight,
    )
    core.write_json(output_dir / "stage1_basis_report.json", stage1_basis_report)
    trained_stage1 = optimize_stage1(
        args,
        selected_ids,
        stage1_bases,
        true_cache,
        reference_cache,
        base_true_nll,
        base_reference_nll,
        masks,
        utility_train_hidden,
        utility_train_probabilities,
        output_dir,
        device=device,
    )
    stage1_delta, stage1_selected, stage1_reports = choose_stage1_delta(
        args,
        trained_stage1,
        stage1_scales,
        true_cache,
        reference_cache,
        prompt_records,
        utility_guard_hidden,
        utility_guard_probabilities,
        device=device,
    )
    torch.save(
        {"row_ids": selected_ids, "delta": stage1_delta.detach().cpu()},
        output_dir / "stage1_delta.pt",
    )
    core.write_json(output_dir / "stage1_scale_reports.json", stage1_reports)
    core.write_json(output_dir / "stage1_selected_report.json", stage1_selected)

    with learner.temporary_materialized_output_delta(
        output_layer, selected_ids, stage1_delta
    ):
        actual_stage1_delta = learner.actual_selected_delta(
            output_layer, selected_ids, base_rows
        )
        stage1_materialized = official_materialized_report(
            model,
            tok,
            originals,
            prompt_records,
            device,
            llama_like=llama_like,
            required_margin=args.required_pairwise_margin,
        )
    add_utility_report(
        stage1_materialized,
        actual_stage1_delta,
        utility_guard_hidden,
        utility_guard_probabilities,
        args,
        device=device,
    )
    core.write_json(output_dir / "stage1_materialized_report.json", stage1_materialized)

    final_delta: torch.Tensor | None = None
    selected_metadata: Dict[str, Any]
    if (
        stage1_materialized["FS"] == 100.0
        and stage1_materialized["GFS"] == 100.0
        and stage1_materialized["direct_margin_failures"] == 0
        and stage1_materialized["paraphrase_margin_failures"] == 0
        and stage1_materialized["utility_safe"]
    ):
        final_delta = actual_stage1_delta
        selected_metadata = {
            "selection_mode": "joint_target_aware_stage1_materialized_FS100_GFS100",
            "stage1": stage1_materialized,
        }
    else:
        failure_positions = stage1_materialized["pairwise_margin_failure_positions"]
        if not failure_positions:
            core.write_json(
                output_dir / "infeasible.json",
                {
                    "method": METHOD,
                    "protocol": PROTOCOL,
                    "stage1": stage1_materialized,
                    "reason": (
                        "Stage 1 met every FS/GFS margin but its checkpoint-dtype "
                        "delta failed a utility guard; no behavioral residual is "
                        "authorized to conceal a utility-only failure"
                    ),
                },
            )
            raise RuntimeError(
                "Stage-1 target-aware delta is behaviorally complete but utility-unsafe"
            )
        failure_set = set(int(value) for value in failure_positions)
        true_failure_mask = torch.tensor(
            [case.record_position in failure_set for case in true_cases],
            device=device,
            dtype=torch.bool,
        )
        reference_failure_mask = torch.tensor(
            [case.record_position in failure_set for case in reference_cases],
            device=device,
            dtype=torch.bool,
        )
        active_ids = sorted(
            set(int(value) for value in true_ids[true_failure_mask].cpu().tolist())
            | set(
                int(value)
                for value in reference_ids[reference_failure_mask].cpu().tolist()
            )
        )
        attempts: List[Dict[str, Any]] = []
        chosen: Dict[str, Any] | None = None
        for rank in rank_ladder:
            # Use every direct/paraphrase predictor state when constructing the
            # active-row bases. The hard solver constraints still decide what
            # may move; the extra contexts provide the rank needed to separate
            # failed prompts from already-successful prompts sharing a row.
            bases, basis_report = build_joint_bases(
                true_hidden,
                true_ids,
                reference_hidden,
                reference_ids,
                utility_second_moment,
                requested_ids=active_ids,
                rank_cap=rank,
                relative_eps=args.contrastive_eps,
                constraint_context_weight=args.constraint_context_weight,
            )
            core.write_json(
                output_dir / f"stage2_rank{rank}_basis_report.json", basis_report
            )
            for solver_target in solver_margins:
                residual, history, solver_report = solve_residual(
                    args,
                    rank=rank,
                    solver_target=solver_target,
                    row_bases=bases,
                    active_ids=active_ids,
                    selected_ids=selected_ids,
                    stage1_delta=actual_stage1_delta,
                    true_cache=true_cache,
                    reference_cache=reference_cache,
                    prompt_records=prompt_records,
                    utility_hidden=utility_train_hidden,
                    utility_probabilities=utility_train_probabilities,
                )
                tag = str(solver_target).replace(".", "p")
                core.write_json(
                    output_dir / f"stage2_rank{rank}_margin{tag}_solver_history.json",
                    history,
                )
                combined = learner.total_delta_with_residual(
                    actual_stage1_delta, selected_ids, residual, active_ids
                )
                if not bool(solver_report["continuous_feasible"]):
                    materialized = {
                        "rank": rank,
                        "solver_target": float(solver_target),
                        "continuous_feasible": False,
                        "feasible": False,
                        "materialization_skipped": True,
                        "skip_reason": (
                            "the continuous candidate failed a hard pairwise, "
                            "Wikipedia, or norm constraint"
                        ),
                        "continuous_FS": solver_report["FS"],
                        "continuous_GFS": solver_report["GFS"],
                        "continuous_minimum_separation": solver_report[
                            "minimum_overall_separation"
                        ],
                        "continuous_total_delta_norm": solver_report[
                            "total_delta_norm"
                        ],
                    }
                    core.write_json(
                        output_dir
                        / f"stage2_rank{rank}_margin{tag}_materialized_report.json",
                        materialized,
                    )
                    attempts.append(
                        {
                            "rank": rank,
                            "solver_target": float(solver_target),
                            "solver": solver_report,
                            "materialized": materialized,
                        }
                    )
                    continue
                with learner.temporary_materialized_output_delta(
                    output_layer, selected_ids, combined
                ):
                    actual_combined = learner.actual_selected_delta(
                        output_layer, selected_ids, base_rows
                    )
                    materialized = official_materialized_report(
                        model,
                        tok,
                        originals,
                        prompt_records,
                        device,
                        llama_like=llama_like,
                        required_margin=args.required_pairwise_margin,
                    )
                add_utility_report(
                    materialized,
                    actual_combined,
                    utility_guard_hidden,
                    utility_guard_probabilities,
                    args,
                    device=device,
                )
                materialized.update(
                    {
                        "rank": rank,
                        "solver_target": float(solver_target),
                        "residual_delta_norm": float(residual.norm().cpu()),
                        "continuous_feasible": bool(
                            solver_report["continuous_feasible"]
                        ),
                    }
                )
                materialized["feasible"] = bool(
                    materialized["continuous_feasible"]
                    and materialized["FS"] == 100.0
                    and materialized["GFS"] == 100.0
                    and materialized["direct_margin_failures"] == 0
                    and materialized["paraphrase_margin_failures"] == 0
                    and materialized["utility_safe"]
                )
                core.write_json(
                    output_dir
                    / f"stage2_rank{rank}_margin{tag}_materialized_report.json",
                    materialized,
                )
                attempt = {
                    "rank": rank,
                    "solver_target": float(solver_target),
                    "solver": solver_report,
                    "materialized": materialized,
                }
                attempts.append(attempt)
                if materialized["feasible"]:
                    chosen = {
                        **attempt,
                        "delta": actual_combined.detach(),
                    }
                    break
            if chosen is not None:
                break
        core.write_json(
            output_dir / "stage2_attempts.json",
            [
                {
                    "rank": row["rank"],
                    "solver_target": row["solver_target"],
                    "solver": row["solver"],
                    "materialized": row["materialized"],
                }
                for row in attempts
            ],
        )
        if chosen is None:
            core.write_json(
                output_dir / "infeasible.json",
                {
                    "method": METHOD,
                    "protocol": PROTOCOL,
                    "stage1": stage1_materialized,
                    "active_row_ids": active_ids,
                    "stage2_attempts": attempts,
                    "reason": (
                        "no checkpoint-dtype candidate achieved FS=100 and GFS=100 "
                        "with positive margins under the locked Wikipedia/norm guards"
                    ),
                },
            )
            raise RuntimeError(
                "Target-aware GA/GD did not find a BF16-safe FS=100/GFS=100 checkpoint"
            )
        final_delta = chosen["delta"]
        selected_metadata = {
            "selection_mode": "exact_residual_materialized_FS100_GFS100",
            "rank": chosen["rank"],
            "solver_target": chosen["solver_target"],
            "solver": chosen["solver"],
            "materialized": chosen["materialized"],
        }

    if final_delta is None:
        raise AssertionError("target-aware selection produced no final delta")
    core.materialize_output_delta(output_layer, selected_ids, final_delta)
    actual_final_delta = learner.actual_selected_delta(
        output_layer, selected_ids, base_rows
    )
    final_report = official_materialized_report(
        model,
        tok,
        originals,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=args.required_pairwise_margin,
    )
    add_utility_report(
        final_report,
        actual_final_delta,
        utility_guard_hidden,
        utility_guard_probabilities,
        args,
        device=device,
    )
    if not (
        final_report["FS"] == 100.0
        and final_report["GFS"] == 100.0
        and final_report["direct_margin_failures"] == 0
        and final_report["paraphrase_margin_failures"] == 0
        and final_report["utility_safe"]
    ):
        raise RuntimeError("final materialized checkpoint failed the FS/GFS guarantee")

    learner.save_checkpoint(model, tok, output_dir / "checkpoint")
    torch.save(
        {"row_ids": selected_ids, "delta": actual_final_delta.detach().cpu()},
        output_dir / "final_total_delta.pt",
    )
    core.write_json(output_dir / "final_FS_GFS_report.json", final_report)
    architecture = {
        "method": METHOD,
        "protocol": PROTOCOL,
        "benchmark_aware": True,
        "editable_parameters": "union_target_true_target_new_lm_head_rows_only",
        "target_true_used_for_bounded_GA": True,
        "target_new_used_for_bounded_GD": True,
        "official_paraphrases_used_for_training_and_checkpoint_selection": True,
        "neighborhood_prompts_used": False,
        "benchmark_retain_examples_used": 0,
        "ppl_text_used": False,
        "required_materialized_FS": 100.0,
        "required_materialized_GFS": 100.0,
        "required_pairwise_margin": float(args.required_pairwise_margin),
        "stage2_solver_margins": list(solver_margins),
        "stage2_rank_ladder": list(rank_ladder),
        "utility_guard_budgets": {
            "mean": float(args.utility_kl_mean_budget),
            "p95": float(args.utility_kl_p95_budget),
            "max": float(args.utility_kl_max_budget),
            "total_delta_norm": float(args.max_total_delta_norm),
        },
    }
    core.write_json(
        output_dir / "config_used.json",
        {
            "schema_version": 1,
            **architecture,
            "architecture_sha256": sha256_bytes(
                json.dumps(architecture, sort_keys=True).encode("utf-8")
            ),
            "seed": int(args.seed),
            "source_mcf_sha256": manifest.get("source_sha256"),
            "forget_case_ids": manifest.get("sampling", {}).get("forget_case_ids", []),
            "selected_row_ids": selected_ids,
            "selected": selected_metadata,
            "final": final_report,
            "utility_cache": str(Path(args.utility_cache).resolve()),
            "utility_cache_metadata": utility_metadata,
        },
    )
    print("Target-aware true-GA/new-GD SURE complete:", output_dir)
    print("Materialized FS:", final_report["FS"])
    print("Materialized GFS:", final_report["GFS"])
    print("Minimum direct separation:", final_report["minimum_direct_separation"])
    print(
        "Minimum paraphrase separation:",
        final_report["minimum_paraphrase_separation"],
    )


if __name__ == "__main__":
    main()
