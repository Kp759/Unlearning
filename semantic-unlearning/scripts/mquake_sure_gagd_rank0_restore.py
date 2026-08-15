#!/usr/bin/env python3
"""SURE MQuAKE: GA/GD -> unrestricted rank-0 forgetting -> low-rank restoration.

This script is the post-GA/GD part of the locked ZeroUnlearn-style MQuAKE
experiment.  It consumes only the training-visible direct rewrite prompts from
50 sampled forget instances.  The 1000 retain instances, atomic questions,
multihop questions, and PPL corpus remain unopened until the caller freezes and
evaluates each candidate.

Stage 1B (rank-0 residual forgetting)
  * load the frozen no-neutral GA/GD checkpoint;
  * untie/freeze the LM head;
  * edit only sensitive target-token rows;
  * each edited row has an unrestricted hidden-size delta (rank 0 in SURE's
    active-repair terminology);
  * enforce every direct sensitive token below the best non-sensitive token by
    a training-only margin.

Stage 2 (utility restoration)
  * build a numerical forget hidden-state basis B_F from all direct forget
    token states;
  * build an ordered restoration basis from non-answer hidden states of the
    same training-visible direct prompts after projection into B_F^perp;
  * branch independently from the same rank-0 solution for rank 64 and 128;
  * project the Base-row restoration into the selected restoration basis;
  * choose the largest restoration scale that preserves all direct-forget
    constraints.  No benchmark retain/AtomicGen/PPL metric is used.

The script saves temporary candidate checkpoints because the repository's
official evaluator consumes model directories.  The launcher deletes those
checkpoints after evaluation while retaining JSON diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import gagd_compare as gagd
import mquake_forget_only_no_neutral as locked
import mquake_zero_unlearn_official_eval as mq


PROTOCOL = "mquake_zerounlearn_forget_only_locked_no_neutral"
METHOD = "SURE-MQuAKE-GAGD-Rank0-LowRankRestore"
DEFAULT_MARGIN_SCHEDULE = (0.25, 0.5, 1.0, 2.0)
DEFAULT_RESTORE_RANKS = (64, 128)
DEFAULT_RESTORE_SCALES = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625, 0.0)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty float list")
    return values


def parse_int_list(text: str) -> Tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("restore ranks must be positive integers")
    return values


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-model-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)

    p.add_argument("--rank0-margin-schedule", type=parse_float_list, default=DEFAULT_MARGIN_SCHEDULE)
    p.add_argument("--rank0-steps-per-margin", type=int, default=600)
    p.add_argument("--rank0-lr", type=float, default=5e-3)
    p.add_argument("--rank0-hard-weight", type=float, default=2.0)
    p.add_argument("--rank0-l2", type=float, default=1e-4)
    p.add_argument("--selection-margin", type=float, default=0.10)

    p.add_argument("--restore-ranks", type=parse_int_list, default=DEFAULT_RESTORE_RANKS)
    p.add_argument("--restore-scales", type=parse_float_list, default=DEFAULT_RESTORE_SCALES)
    p.add_argument("--geometry-tolerance", type=float, default=5e-5)
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    if a.forget_num != 50:
        raise ValueError("this locked experiment fixes --forget-num=50")
    if a.cache_batch_size <= 0 or a.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if a.rank0_steps_per_margin <= 0:
        raise ValueError("--rank0-steps-per-margin must be positive")
    if a.rank0_lr <= 0 or a.rank0_hard_weight <= 0:
        raise ValueError("rank-0 optimizer values must be positive")
    if a.rank0_l2 < 0 or a.selection_margin < 0:
        raise ValueError("rank-0 L2 and selection margin must be non-negative")
    if any(x <= 0 for x in a.rank0_margin_schedule):
        raise ValueError("rank-0 margins must be positive")
    if tuple(sorted(set(a.rank0_margin_schedule))) != tuple(a.rank0_margin_schedule):
        raise ValueError("rank-0 margin schedule must be strictly increasing")
    if 0.0 not in a.restore_scales:
        raise ValueError("restore scales must contain 0")
    if any(x < 0 or x > 1 for x in a.restore_scales):
        raise ValueError("restore scales must be in [0,1]")
    if a.geometry_tolerance <= 0:
        raise ValueError("geometry tolerance must be positive")


def _device(model: nn.Module) -> torch.device:
    return gagd.first_device(model)


@torch.no_grad()
def cache_case_states(
    model: nn.Module,
    tok: Any,
    cases: Sequence[mq.PredictionCase],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    model.eval()
    for start in tqdm(range(0, len(cases), batch_size), desc="cache direct forget states"):
        batch = cases[start : start + batch_size]
        enc = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        out = model(**enc, use_cache=False, output_hidden_states=True)
        pos = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden.append(out.hidden_states[-1][idx, pos, :].detach().float())
        logits.append(out.logits[idx, pos, :].detach().float())
    return torch.cat(hidden, 0), torch.cat(logits, 0)


@torch.no_grad()
def cache_prompt_token_hidden(
    model: nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    prompts = []
    for record in records:
        rr = record["requested_rewrite"]
        prompts.append(str(rr["prompt"]).format(str(rr["subject"])))
    rows: List[torch.Tensor] = []
    model.eval()
    for start in tqdm(range(0, len(prompts), batch_size), desc="cache training-visible nonsensitive states"):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, padding=True, return_tensors="pt").to(device)
        out = model(**enc, use_cache=False, output_hidden_states=True)
        h = out.hidden_states[-1].detach().float()
        mask = enc["attention_mask"].bool()
        for i in range(h.shape[0]):
            # Context states only.  The final prompt position is a direct-forget
            # state and is removed by the B_F projection below, but keeping it
            # here makes the construction deterministic and auditable.
            rows.append(h[i, mask[i], :])
    if not rows:
        raise RuntimeError("no training-visible prompt hidden states were cached")
    return torch.cat(rows, 0)


def row_basis(rows: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    values = rows.float()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("row basis requires a non-empty matrix")
    _, s, vh = torch.linalg.svd(values, full_matrices=False)
    tol = max(values.shape) * torch.finfo(torch.float32).eps * s.max().clamp_min(1.0)
    rank = int((s > tol).sum().item())
    basis = vh[:rank].contiguous()
    if rank:
        q, _ = torch.linalg.qr(basis.T, mode="reduced")
        basis = q.T.contiguous()
    report = {
        "state_count": int(values.shape[0]),
        "hidden_size": int(values.shape[1]),
        "numerical_rank": rank,
        "complement_dimension": int(values.shape[1] - rank),
        "svd_tolerance": float(tol.item()),
    }
    return basis, report


def project_away(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    x = rows.float()
    if basis.numel() == 0:
        return x
    b = basis.float()
    return x - (x @ b.T) @ b


def build_restore_basis(
    utility_hidden: torch.Tensor,
    forget_basis: torch.Tensor,
    max_rank: int,
    geometry_tolerance: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    null_hidden = project_away(utility_hidden, forget_basis)
    basis, report = row_basis(null_hidden)
    if basis.shape[0] < max_rank:
        raise RuntimeError(
            f"training-visible forget-complement basis rank {basis.shape[0]} is below requested rank {max_rank}"
        )
    basis = basis[:max_rank].contiguous()
    # One more projection/QR removes accumulated SVD/FP32 overlap with B_F.
    basis = project_away(basis, forget_basis)
    q, _ = torch.linalg.qr(basis.T, mode="reduced")
    basis = q.T[:max_rank].contiguous()
    overlap = basis @ forget_basis.T if forget_basis.numel() else basis.new_zeros((max_rank, 0))
    gram = basis @ basis.T
    eye = torch.eye(max_rank, device=basis.device, dtype=basis.dtype)
    max_overlap = float(overlap.abs().max().item()) if overlap.numel() else 0.0
    orth_error = float((gram - eye).abs().max().item())
    if max_overlap > geometry_tolerance:
        raise RuntimeError(f"restoration basis leaves forget complement: {max_overlap}")
    if orth_error > geometry_tolerance:
        raise RuntimeError(f"restoration basis is not orthonormal: {orth_error}")
    report.update(
        {
            "requested_max_rank": int(max_rank),
            "returned_rank": int(basis.shape[0]),
            "maximum_forget_overlap": max_overlap,
            "orthogonality_max_error": orth_error,
        }
    )
    return basis, report


def _selected_target_indices(target_ids: torch.Tensor, row_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    mapping = {int(token_id): i for i, token_id in enumerate(row_ids)}
    result = [mapping[int(token_id)] for token_id in target_ids.detach().cpu().tolist()]
    return torch.tensor(result, dtype=torch.long, device=device)


def _best_nonsensitive_logits(logits: torch.Tensor, row_ids: Sequence[int]) -> torch.Tensor:
    masked = logits.float().clone()
    ids = torch.tensor(row_ids, dtype=torch.long, device=masked.device)
    masked.index_fill_(1, ids, -torch.inf)
    return masked.max(dim=1).values


def margins_from_delta(
    hidden: torch.Tensor,
    target_stage1_logits: torch.Tensor,
    best_nonsensitive: torch.Tensor,
    target_row_index: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    target_delta = delta.index_select(0, target_row_index)
    effect = (hidden.float() * target_delta.float()).sum(dim=1)
    target = target_stage1_logits.float() + effect
    return best_nonsensitive.float() - target


def optimize_rank0(
    hidden: torch.Tensor,
    target_stage1_logits: torch.Tensor,
    best_nonsensitive: torch.Tensor,
    target_row_index: torch.Tensor,
    row_count: int,
    hidden_size: int,
    *,
    margin_schedule: Sequence[float],
    steps_per_margin: int,
    lr: float,
    hard_weight: float,
    l2: float,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    delta = nn.Parameter(torch.zeros((row_count, hidden_size), device=hidden.device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([delta], lr=lr, weight_decay=0.0)
    log: List[Dict[str, Any]] = []
    for required in margin_schedule:
        for step in range(1, steps_per_margin + 1):
            optimizer.zero_grad(set_to_none=True)
            margins = margins_from_delta(hidden, target_stage1_logits, best_nonsensitive, target_row_index, delta)
            violation = torch.relu(float(required) - margins)
            count = int((violation > 0).sum().item())
            if count == 0:
                break
            hard = violation.max().square()
            mean = violation.square().mean()
            reg = delta.square().sum()
            loss = hard_weight * hard + mean + l2 * reg
            if not torch.isfinite(loss):
                raise FloatingPointError("rank-0 loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], 1.0)
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == steps_per_margin:
                after = margins_from_delta(hidden, target_stage1_logits, best_nonsensitive, target_row_index, delta.detach())
                log.append(
                    {
                        "required_margin": float(required),
                        "step": int(step),
                        "violation_count": int((after < required).sum().item()),
                        "minimum_margin": float(after.min().item()),
                        "mean_margin": float(after.mean().item()),
                        "delta_norm": float(delta.detach().norm().item()),
                        "loss": float(loss.detach().item()),
                    }
                )
        final = margins_from_delta(hidden, target_stage1_logits, best_nonsensitive, target_row_index, delta.detach())
        log.append(
            {
                "required_margin": float(required),
                "phase_complete": True,
                "violation_count": int((final < required).sum().item()),
                "minimum_margin": float(final.min().item()),
                "mean_margin": float(final.mean().item()),
                "delta_norm": float(delta.detach().norm().item()),
            }
        )
        if int((final < required).sum().item()) != 0:
            raise RuntimeError(f"rank-0 phase failed to satisfy margin {required}")
    return delta.detach().float(), log


@torch.no_grad()
def set_rows(weight: torch.Tensor, row_ids: Sequence[int], rows: torch.Tensor) -> None:
    ids = torch.tensor(row_ids, dtype=torch.long, device=weight.device)
    weight.index_copy_(0, ids, rows.to(device=weight.device, dtype=weight.dtype))


def load_base_rows(base_model_path: str, row_ids: Sequence[int]) -> torch.Tensor:
    # Load on CPU only long enough to snapshot the selected output rows.
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    ids = torch.tensor(row_ids, dtype=torch.long, device=base.get_output_embeddings().weight.device)
    rows = base.get_output_embeddings().weight.index_select(0, ids).detach().float().cpu()
    del base
    gc.collect()
    return rows


def direct_correct_count(
    model: nn.Module,
    tok: Any,
    cases: Sequence[mq.PredictionCase],
    llama_like: bool,
    device: torch.device,
    batch_size: int,
) -> int:
    rows = locked.predict(model, tok, cases, llama_like, device, batch_size)
    return sum(int(row["correct"]) for row in rows)


def main() -> None:
    a = args()
    validate(a)
    gagd.set_seed(a.seed)

    root = Path(a.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    visible = Path(a.training_visible_path).resolve()
    manifest = Path(a.split_manifest).resolve()
    records, split = locked.load_locked(visible, manifest, a.seed, a.forget_num)

    ns = argparse.Namespace(
        model_path=a.stage1_model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = _device(model)
    llama_like = mq.is_llama_like(model, tok)
    out = locked.untie(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    cases = locked.all_cases(records, tok, llama_like)
    if not cases:
        raise RuntimeError("locked MQuAKE split produced zero direct token cases")
    hidden, logits = cache_case_states(model, tok, cases, device, a.cache_batch_size)
    target_ids = locked.target_ids(tok, cases, llama_like, device)
    row_ids = sorted(set(int(x) for x in target_ids.detach().cpu().tolist()))
    target_row_index = _selected_target_indices(target_ids, row_ids, device)
    row_id_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
    stage1_anchor_rows = out.weight.index_select(0, row_id_tensor).detach().float()
    target_stage1_logits = logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    best_nonsensitive = _best_nonsensitive_logits(logits, row_ids)
    stage1_margins = best_nonsensitive - target_stage1_logits

    base_rows_cpu = load_base_rows(a.base_model_path, row_ids)
    base_rows = base_rows_cpu.to(device=device, dtype=torch.float32)

    rank0_delta, rank0_log = optimize_rank0(
        hidden,
        target_stage1_logits,
        best_nonsensitive,
        target_row_index,
        len(row_ids),
        int(out.weight.shape[1]),
        margin_schedule=a.rank0_margin_schedule,
        steps_per_margin=a.rank0_steps_per_margin,
        lr=a.rank0_lr,
        hard_weight=a.rank0_hard_weight,
        l2=a.rank0_l2,
    )
    rank0_rows_fp32 = stage1_anchor_rows + rank0_delta
    set_rows(out.weight, row_ids, rank0_rows_fp32)

    # Re-cache after BF16 materialization.  Hidden states are unchanged by an
    # output-only repair, but using the materialized logits closes the FP32/BF16
    # audit loop before restoration.
    rank0_hidden, rank0_logits = cache_case_states(model, tok, cases, device, a.cache_batch_size)
    rank0_target_logits = rank0_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    rank0_best_nonsensitive = _best_nonsensitive_logits(rank0_logits, row_ids)
    rank0_margins_actual = rank0_best_nonsensitive - rank0_target_logits
    rank0_correct = direct_correct_count(model, tok, cases, llama_like, device, a.eval_batch_size)
    if rank0_correct != 0:
        raise RuntimeError(f"rank-0 materialized checkpoint still has {rank0_correct} direct sensitive top-1 cases")
    if float(rank0_margins_actual.min().item()) < a.selection_margin:
        raise RuntimeError(
            "rank-0 materialized minimum margin is below the fixed restoration selection margin: "
            f"{float(rank0_margins_actual.min().item())} < {a.selection_margin}"
        )

    forget_basis, forget_geometry = row_basis(rank0_hidden)
    utility_hidden = cache_prompt_token_hidden(model, tok, records, device, a.cache_batch_size)
    max_restore_rank = max(a.restore_ranks)
    restore_basis, restore_geometry = build_restore_basis(
        utility_hidden,
        forget_basis,
        max_restore_rank,
        a.geometry_tolerance,
    )

    rank0_rows_actual = out.weight.index_select(0, row_id_tensor).detach().float()
    fixed_base_delta = rank0_rows_actual - base_rows
    rank0_utility_mse = float((utility_hidden @ fixed_base_delta.T).square().mean().item())
    rank0_weight_distance = float(fixed_base_delta.norm().item())

    per_rank: Dict[str, Any] = {}
    sorted_scales = sorted(set(float(x) for x in a.restore_scales), reverse=True)
    for rank in a.restore_ranks:
        basis = restore_basis[:rank].contiguous()
        # Best row-space restoration toward Base within span(B_R).
        desired = base_rows - rank0_rows_actual
        coefficients = desired @ basis.T
        full_restore = coefficients @ basis
        h_rank = rank0_hidden @ basis.T
        target_coeff = coefficients.index_select(0, target_row_index)
        full_target_effect = (h_rank * target_coeff).sum(dim=1)

        scale_reports = []
        selected_scale = None
        for scale in sorted_scales:
            candidate_margins = rank0_margins_actual - float(scale) * full_target_effect
            violations = int((candidate_margins < a.selection_margin).sum().item())
            report = {
                "scale": float(scale),
                "violation_count": violations,
                "minimum_direct_margin": float(candidate_margins.min().item()),
                "mean_direct_margin": float(candidate_margins.mean().item()),
            }
            scale_reports.append(report)
            if selected_scale is None and violations == 0:
                selected_scale = float(scale)
        if selected_scale is None:
            raise RuntimeError(f"rank-{rank} restoration found no forget-feasible scale including zero")

        candidate_rows = rank0_rows_actual + selected_scale * full_restore
        set_rows(out.weight, row_ids, candidate_rows)
        correct = direct_correct_count(model, tok, cases, llama_like, device, a.eval_batch_size)
        if correct != 0:
            raise RuntimeError(f"rank-{rank} BF16 materialization reactivated {correct} direct sensitive cases")

        _, candidate_logits = cache_case_states(model, tok, cases, device, a.cache_batch_size)
        candidate_target = candidate_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
        candidate_best = _best_nonsensitive_logits(candidate_logits, row_ids)
        candidate_margin = candidate_best - candidate_target
        if float(candidate_margin.min().item()) < a.selection_margin:
            raise RuntimeError(
                f"rank-{rank} BF16 candidate violates selection margin after materialization: "
                f"{float(candidate_margin.min().item())}"
            )

        candidate_delta_to_base = candidate_rows - base_rows
        utility_mse = float((utility_hidden @ candidate_delta_to_base.T).square().mean().item())
        summary = {
            "rank": int(rank),
            "selected_scale": selected_scale,
            "direct_correct_sensitive_tokens": int(correct),
            "minimum_direct_margin": float(candidate_margin.min().item()),
            "mean_direct_margin": float(candidate_margin.mean().item()),
            "rank0_utility_selected_logit_mse": rank0_utility_mse,
            "candidate_utility_selected_logit_mse": utility_mse,
            "rank0_selected_row_distance_to_base": rank0_weight_distance,
            "candidate_selected_row_distance_to_base": float(candidate_delta_to_base.norm().item()),
            "full_restore_norm": float(full_restore.norm().item()),
            "effective_restore_norm": float((selected_scale * full_restore).norm().item()),
            "scale_reports": scale_reports,
            "benchmark_retain_seen": 0,
            "atomic_questions_seen": 0,
            "multihop_questions_seen": 0,
            "PPL_seen": False,
            "selection_uses_only_training_visible_direct_forget_constraints": True,
        }
        rank_dir = root / f"rank{rank}"
        if a.save_model:
            checkpoint = rank_dir / "checkpoint"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint)
            tok.save_pretrained(checkpoint)
            summary["checkpoint"] = str(checkpoint.resolve())
        write(rank_dir / "restore_summary.json", summary)
        per_rank[str(rank)] = summary

        # Independent branch: every restoration rank starts from the exact same
        # materialized rank-0 rows, never from another restoration candidate.
        set_rows(out.weight, row_ids, rank0_rows_actual)

    overall = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": int(len(records)),
        "direct_sensitive_token_cases": int(len(cases)),
        "sensitive_output_rows": int(len(row_ids)),
        "stage1_checkpoint": str(Path(a.stage1_model_path).resolve()),
        "base_model": str(Path(a.base_model_path).resolve()),
        "rank0": {
            "parameterization": "unrestricted selected sensitive LM-head rows; no low-rank factorization",
            "row_shape": [int(len(row_ids)), int(out.weight.shape[1])],
            "stage1_minimum_margin_before": float(stage1_margins.min().item()),
            "materialized_minimum_margin": float(rank0_margins_actual.min().item()),
            "materialized_mean_margin": float(rank0_margins_actual.mean().item()),
            "direct_correct_sensitive_tokens": int(rank0_correct),
            "delta_norm": float(rank0_delta.norm().item()),
            "margin_schedule": [float(x) for x in a.rank0_margin_schedule],
            "log": rank0_log,
        },
        "forget_geometry": forget_geometry,
        "restore_geometry": restore_geometry,
        "restoration_ranks": per_rank,
        "data_firewall": {
            "training_visible": "same 50 sampled forget instances, direct requested_rewrite only",
            "benchmark_retain_seen_before_freeze": 0,
            "atomic_questions_seen_before_freeze": 0,
            "multihop_questions_seen_before_freeze": 0,
            "PPL_seen_before_freeze": False,
            "target_new_seen": False,
            "Unknown_or_IDK_used": False,
        },
        "split_sampling": split.get("sampling"),
    }
    write(root / "summary.json", overall)
    print("===== SURE MQuAKE GA/GD -> rank0 -> restore =====")
    print("rank0 direct correct:", rank0_correct)
    print("rank0 min margin:", float(rank0_margins_actual.min().item()))
    for rank in a.restore_ranks:
        r = per_rank[str(rank)]
        print(
            f"rank{rank}: scale={r['selected_scale']} direct_correct={r['direct_correct_sensitive_tokens']} "
            f"min_margin={r['minimum_direct_margin']:.6f} utility_mse={r['candidate_utility_selected_logit_mse']:.6f}"
        )
    print("summary:", root / "summary.json")


if __name__ == "__main__":
    main()
