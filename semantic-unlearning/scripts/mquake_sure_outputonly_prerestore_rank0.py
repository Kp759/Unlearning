#!/usr/bin/env python3
"""SURE MQuAKE with output-only Stage1 localization and pre-rank0 row restoration.

Protocol firewall
-----------------
Only the locked 50 training-visible MQuAKE forget instances are consumed here.
The 1000 benchmark-retain instances, held-out AtomicGen questions, multihop
questions, and PPL corpus are never opened by this script.

Architecture
------------
1. Load a frozen Stage1 GA/GD checkpoint created with restoration_mode=output_only:
   every input-embedding row is Base, every non-sensitive LM-head row is Base,
   and only sensitive LM-head output rows retain Stage1 displacement.
2. Build the direct-forget hidden-state span B_F from the same 50 visible facts.
3. For each sensitive LM-head row, decompose its Stage1 displacement from Base
   into a component in span(B_F) and a component in span(B_F)^perp. Restore the
   perpendicular component toward Base. The largest predefined scale is chosen
   using direct-forget constraints only; no held-out utility metric is visible.
4. From that frozen pre-restored anchor, run unrestricted selected-row rank-0
   repair to enforce the fixed direct-forget margin.
5. Independently branch rank-64 and rank-128 utility restoration from the same
   materialized rank-0 solution, exactly as in the previous SURE experiment.

The pre-rank0 restoration is deliberately conservative: a candidate is allowed
only if it does not reactivate a direct case that was already non-top1 at the
Stage1 anchor and does not decrease any direct nonsensitive margin by more than
an explicitly numerical tolerance. Rank-0 then closes any residual Stage1
forget failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

import gagd_compare as gagd
import mquake_forget_only_no_neutral as locked
import mquake_zero_unlearn_official_eval as mq
import mquake_sure_gagd_rank0_restore as core


METHOD = "SURE-MQuAKE-OutputOnly-PreRestore-Rank0-LowRankRestore"
DEFAULT_SCALES = core.DEFAULT_RESTORE_SCALES


def parse_args() -> argparse.Namespace:
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

    p.add_argument("--pre-rank0-restore-scales", type=core.parse_float_list, default=DEFAULT_SCALES)
    p.add_argument(
        "--pre-rank0-margin-tolerance",
        type=float,
        default=0.01,
        help="maximum BF16 numerical decrease allowed in any Stage1 direct nonsensitive margin",
    )
    p.add_argument("--require-stage1-output-only", action="store_true")
    p.add_argument("--save-pre-rank0-model", action="store_true")

    p.add_argument("--rank0-margin-schedule", type=core.parse_float_list, default=(0.25,))
    p.add_argument("--rank0-steps-per-margin", type=int, default=600)
    p.add_argument("--rank0-lr", type=float, default=5e-3)
    p.add_argument("--rank0-hard-weight", type=float, default=2.0)
    p.add_argument("--rank0-l2", type=float, default=1e-4)
    p.add_argument("--selection-margin", type=float, default=0.10)

    p.add_argument("--restore-ranks", type=core.parse_int_list, default=core.DEFAULT_RESTORE_RANKS)
    p.add_argument("--restore-scales", type=core.parse_float_list, default=DEFAULT_SCALES)
    p.add_argument("--geometry-tolerance", type=float, default=5e-5)
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    if a.forget_num != 50:
        raise ValueError("this locked experiment fixes --forget-num=50")
    if a.cache_batch_size <= 0 or a.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if a.pre_rank0_margin_tolerance < 0:
        raise ValueError("pre-rank0 margin tolerance must be non-negative")
    for values, name in (
        (a.pre_rank0_restore_scales, "pre-rank0 restore scales"),
        (a.restore_scales, "restore scales"),
    ):
        if 0.0 not in values:
            raise ValueError(f"{name} must contain 0")
        if any(x < 0 or x > 1 for x in values):
            raise ValueError(f"{name} must lie in [0,1]")
    if a.rank0_steps_per_margin <= 0 or a.rank0_lr <= 0 or a.rank0_hard_weight <= 0:
        raise ValueError("rank-0 optimizer values must be positive")
    if a.rank0_l2 < 0 or a.selection_margin < 0:
        raise ValueError("rank-0 L2 and selection margin must be non-negative")
    if any(x <= 0 for x in a.rank0_margin_schedule):
        raise ValueError("rank-0 margins must be positive")
    if tuple(sorted(set(a.rank0_margin_schedule))) != tuple(a.rank0_margin_schedule):
        raise ValueError("rank-0 margin schedule must be strictly increasing")
    if a.geometry_tolerance <= 0:
        raise ValueError("geometry tolerance must be positive")


def stage1_config(stage1_model_path: str) -> Dict[str, Any]:
    ckpt = Path(stage1_model_path).resolve()
    config_path = ckpt.parent / "config_used.json"
    if not config_path.is_file():
        return {"config_found": False, "config_path": str(config_path)}
    value = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "config_found": True,
        "config_path": str(config_path),
        "restoration_mode": value.get("restoration_mode"),
        "retain_seen": value.get("retain_seen"),
        "atomic_questions_seen": value.get("atomic_questions_seen"),
        "multihop_questions_seen": value.get("multihop_questions_seen"),
        "PPL_seen": value.get("PPL_seen"),
        "target_new_seen": value.get("target_new_seen"),
    }


@torch.no_grad()
def choose_pre_rank0_restore(
    model,
    tok,
    cases: Sequence[mq.PredictionCase],
    device: torch.device,
    target_ids: torch.Tensor,
    row_ids: Sequence[int],
    stage1_rows: torch.Tensor,
    base_rows: torch.Tensor,
    hidden: torch.Tensor,
    stage1_logits: torch.Tensor,
    forget_basis: torch.Tensor,
    scales: Sequence[float],
    margin_tolerance: float,
    batch_size: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Restore Stage1 sensitive rows toward Base outside the direct-forget span."""
    out = model.get_output_embeddings()
    stage1_target = stage1_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    stage1_best = core._best_nonsensitive_logits(stage1_logits, row_ids)
    stage1_margins = stage1_best - stage1_target
    stage1_top1 = stage1_logits.argmax(dim=-1).eq(target_ids)

    displacement = stage1_rows - base_rows
    if forget_basis.numel():
        forget_component = (displacement @ forget_basis.T) @ forget_basis
    else:
        forget_component = torch.zeros_like(displacement)
    collateral_component = displacement - forget_component
    projected_rows = base_rows + forget_component
    full_restore = projected_rows - stage1_rows

    reports = []
    selected_scale = None
    selected_rows = None
    selected_logits = None
    sorted_scales = sorted(set(float(x) for x in scales), reverse=True)
    for scale in sorted_scales:
        candidate_rows = stage1_rows + float(scale) * full_restore
        core.set_rows(out.weight, row_ids, candidate_rows)
        _, candidate_logits = core.cache_case_states(model, tok, cases, device, batch_size)
        candidate_target = candidate_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
        candidate_best = core._best_nonsensitive_logits(candidate_logits, row_ids)
        candidate_margins = candidate_best - candidate_target
        candidate_top1 = candidate_logits.argmax(dim=-1).eq(target_ids)

        reactivated = int(((~stage1_top1) & candidate_top1).sum().item())
        margin_change = candidate_margins - stage1_margins
        weakened = int((margin_change < -float(margin_tolerance)).sum().item())
        report = {
            "scale": float(scale),
            "reactivated_previously_forgotten_direct_token_cases": reactivated,
            "weakened_direct_margins_beyond_tolerance": weakened,
            "direct_target_top1_cases": int(candidate_top1.sum().item()),
            "minimum_direct_margin": float(candidate_margins.min().item()),
            "mean_direct_margin": float(candidate_margins.mean().item()),
            "minimum_margin_change_from_stage1": float(margin_change.min().item()),
            "maximum_margin_change_from_stage1": float(margin_change.max().item()),
        }
        reports.append(report)
        if selected_scale is None and reactivated == 0 and weakened == 0:
            selected_scale = float(scale)
            selected_rows = out.weight.index_select(
                0, torch.tensor(row_ids, dtype=torch.long, device=out.weight.device)
            ).detach().float()
            selected_logits = candidate_logits.detach().float()

    if selected_scale is None or selected_rows is None or selected_logits is None:
        core.set_rows(out.weight, row_ids, stage1_rows)
        raise RuntimeError("pre-rank0 restoration found no direct-forget-safe scale, including zero")

    core.set_rows(out.weight, row_ids, selected_rows)
    selected_target = selected_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    selected_best = core._best_nonsensitive_logits(selected_logits, row_ids)
    selected_margins = selected_best - selected_target

    summary = {
        "definition": "restore the component of each sensitive Stage1 LM-head displacement lying in the orthogonal complement of the direct-forget hidden span",
        "selected_scale": selected_scale,
        "margin_tolerance": float(margin_tolerance),
        "sensitive_output_rows": int(len(row_ids)),
        "forget_basis_rank": int(forget_basis.shape[0]),
        "stage1_displacement_norm": float(displacement.norm().item()),
        "forget_span_component_norm": float(forget_component.norm().item()),
        "collateral_complement_component_norm": float(collateral_component.norm().item()),
        "full_restore_norm": float(full_restore.norm().item()),
        "effective_restore_norm": float((selected_scale * full_restore).norm().item()),
        "stage1_direct_target_top1_cases": int(stage1_top1.sum().item()),
        "selected_direct_target_top1_cases": int(selected_logits.argmax(dim=-1).eq(target_ids).sum().item()),
        "stage1_minimum_direct_margin": float(stage1_margins.min().item()),
        "selected_minimum_direct_margin": float(selected_margins.min().item()),
        "stage1_mean_direct_margin": float(stage1_margins.mean().item()),
        "selected_mean_direct_margin": float(selected_margins.mean().item()),
        "scale_reports": reports,
        "benchmark_retain_seen": 0,
        "atomic_questions_seen": 0,
        "multihop_questions_seen": 0,
        "PPL_seen": False,
        "selection_uses_only_training_visible_direct_forget_constraints": True,
    }
    return selected_rows, summary


def main() -> None:
    a = parse_args()
    validate(a)
    gagd.set_seed(a.seed)

    root = Path(a.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    visible = Path(a.training_visible_path).resolve()
    manifest = Path(a.split_manifest).resolve()
    records, split = locked.load_locked(visible, manifest, a.seed, a.forget_num)

    s1cfg = stage1_config(a.stage1_model_path)
    if a.require_stage1_output_only:
        if not s1cfg.get("config_found"):
            raise RuntimeError("--require-stage1-output-only set but Stage1 config_used.json was not found")
        if s1cfg.get("restoration_mode") != "output_only":
            raise RuntimeError(
                f"Stage1 checkpoint restoration_mode={s1cfg.get('restoration_mode')!r}, expected 'output_only'"
            )

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
    device = core._device(model)
    llama_like = mq.is_llama_like(model, tok)
    out = locked.untie(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    cases = locked.all_cases(records, tok, llama_like)
    if not cases:
        raise RuntimeError("locked MQuAKE split produced zero direct token cases")

    stage1_hidden, stage1_logits = core.cache_case_states(model, tok, cases, device, a.cache_batch_size)
    target_ids = locked.target_ids(tok, cases, llama_like, device)
    row_ids = sorted(set(int(x) for x in target_ids.detach().cpu().tolist()))
    target_row_index = core._selected_target_indices(target_ids, row_ids, device)
    row_id_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
    stage1_rows = out.weight.index_select(0, row_id_tensor).detach().float()
    base_rows = core.load_base_rows(a.base_model_path, row_ids).to(device=device, dtype=torch.float32)

    # This basis is built before rank-0, from exactly the Stage1 direct-forget
    # hidden states. Output-row restoration does not alter those hidden states.
    pre_forget_basis, pre_forget_geometry = core.row_basis(stage1_hidden)
    pre_rows, pre_summary = choose_pre_rank0_restore(
        model,
        tok,
        cases,
        device,
        target_ids,
        row_ids,
        stage1_rows,
        base_rows,
        stage1_hidden,
        stage1_logits,
        pre_forget_basis,
        a.pre_rank0_restore_scales,
        a.pre_rank0_margin_tolerance,
        a.cache_batch_size,
    )

    # Re-cache the selected BF16 materialization and make it the rank-0 anchor.
    pre_hidden, pre_logits = core.cache_case_states(model, tok, cases, device, a.cache_batch_size)
    pre_target_logits = pre_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    pre_best_nonsensitive = core._best_nonsensitive_logits(pre_logits, row_ids)
    pre_margins = pre_best_nonsensitive - pre_target_logits
    pre_direct_correct = core.direct_correct_count(model, tok, cases, llama_like, device, a.eval_batch_size)
    pre_summary["materialized_direct_correct_sensitive_tokens"] = int(pre_direct_correct)
    pre_summary["materialized_minimum_direct_margin"] = float(pre_margins.min().item())
    pre_summary["materialized_mean_direct_margin"] = float(pre_margins.mean().item())

    pre_dir = root / "pre_rank0"
    core.write(pre_dir / "restore_summary.json", pre_summary)
    if a.save_pre_rank0_model:
        ckpt = pre_dir / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        pre_summary["checkpoint"] = str(ckpt.resolve())
        core.write(pre_dir / "restore_summary.json", pre_summary)

    rank0_delta, rank0_log = core.optimize_rank0(
        pre_hidden,
        pre_target_logits,
        pre_best_nonsensitive,
        target_row_index,
        len(row_ids),
        int(out.weight.shape[1]),
        margin_schedule=a.rank0_margin_schedule,
        steps_per_margin=a.rank0_steps_per_margin,
        lr=a.rank0_lr,
        hard_weight=a.rank0_hard_weight,
        l2=a.rank0_l2,
    )
    rank0_rows_fp32 = pre_rows + rank0_delta
    core.set_rows(out.weight, row_ids, rank0_rows_fp32)

    rank0_hidden, rank0_logits = core.cache_case_states(model, tok, cases, device, a.cache_batch_size)
    rank0_target = rank0_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    rank0_best = core._best_nonsensitive_logits(rank0_logits, row_ids)
    rank0_margins = rank0_best - rank0_target
    rank0_correct = core.direct_correct_count(model, tok, cases, llama_like, device, a.eval_batch_size)
    if rank0_correct != 0:
        raise RuntimeError(f"rank-0 materialized checkpoint still has {rank0_correct} direct sensitive top-1 cases")
    if float(rank0_margins.min().item()) < a.selection_margin:
        raise RuntimeError(
            f"rank-0 minimum margin {float(rank0_margins.min().item())} is below selection margin {a.selection_margin}"
        )

    forget_basis, forget_geometry = core.row_basis(rank0_hidden)
    utility_hidden = core.cache_prompt_token_hidden(model, tok, records, device, a.cache_batch_size)
    max_restore_rank = max(a.restore_ranks)
    restore_basis, restore_geometry = core.build_restore_basis(
        utility_hidden, forget_basis, max_restore_rank, a.geometry_tolerance
    )

    rank0_rows = out.weight.index_select(0, row_id_tensor).detach().float()
    fixed_base_delta = rank0_rows - base_rows
    rank0_utility_mse = float((utility_hidden @ fixed_base_delta.T).square().mean().item())
    rank0_weight_distance = float(fixed_base_delta.norm().item())

    per_rank: Dict[str, Any] = {}
    sorted_scales = sorted(set(float(x) for x in a.restore_scales), reverse=True)
    for rank in a.restore_ranks:
        basis = restore_basis[:rank].contiguous()
        desired = base_rows - rank0_rows
        coefficients = desired @ basis.T
        full_restore = coefficients @ basis
        h_rank = rank0_hidden @ basis.T
        target_coeff = coefficients.index_select(0, target_row_index)
        full_target_effect = (h_rank * target_coeff).sum(dim=1)

        scale_reports = []
        selected_scale = None
        for scale in sorted_scales:
            candidate_margins = rank0_margins - float(scale) * full_target_effect
            violations = int((candidate_margins < a.selection_margin).sum().item())
            scale_reports.append(
                {
                    "scale": float(scale),
                    "violation_count": violations,
                    "minimum_direct_margin": float(candidate_margins.min().item()),
                    "mean_direct_margin": float(candidate_margins.mean().item()),
                }
            )
            if selected_scale is None and violations == 0:
                selected_scale = float(scale)
        if selected_scale is None:
            raise RuntimeError(f"rank-{rank} restoration found no forget-feasible scale including zero")

        candidate_rows = rank0_rows + selected_scale * full_restore
        core.set_rows(out.weight, row_ids, candidate_rows)
        correct = core.direct_correct_count(model, tok, cases, llama_like, device, a.eval_batch_size)
        if correct != 0:
            raise RuntimeError(f"rank-{rank} BF16 materialization reactivated {correct} direct sensitive cases")

        _, candidate_logits = core.cache_case_states(model, tok, cases, device, a.cache_batch_size)
        candidate_target = candidate_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
        candidate_best = core._best_nonsensitive_logits(candidate_logits, row_ids)
        candidate_margin = candidate_best - candidate_target
        if float(candidate_margin.min().item()) < a.selection_margin:
            raise RuntimeError(
                f"rank-{rank} BF16 candidate violates selection margin: {float(candidate_margin.min().item())}"
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
        core.write(rank_dir / "restore_summary.json", summary)
        per_rank[str(rank)] = summary
        core.set_rows(out.weight, row_ids, rank0_rows)

    overall = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": core.PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "direct_sensitive_token_cases": int(len(cases)),
        "sensitive_output_rows": int(len(row_ids)),
        "stage1_checkpoint": str(Path(a.stage1_model_path).resolve()),
        "stage1_config": s1cfg,
        "base_model": str(Path(a.base_model_path).resolve()),
        "pre_rank0_restore": pre_summary,
        "pre_rank0_forget_geometry": pre_forget_geometry,
        "rank0": {
            "parameterization": "unrestricted selected sensitive LM-head rows; no low-rank factorization",
            "anchor": "output-only Stage1 after forget-span complement restoration",
            "row_shape": [int(len(row_ids)), int(out.weight.shape[1])],
            "anchor_minimum_margin": float(pre_margins.min().item()),
            "materialized_minimum_margin": float(rank0_margins.min().item()),
            "materialized_mean_margin": float(rank0_margins.mean().item()),
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
    core.write(root / "summary.json", overall)

    print("===== SURE MQuAKE output-only -> pre-restore -> rank0 -> restore =====")
    print(
        "pre-rank0: scale=",
        pre_summary["selected_scale"],
        "stage1_direct_top1=",
        pre_summary["stage1_direct_target_top1_cases"],
        "selected_direct_top1=",
        pre_summary["selected_direct_target_top1_cases"],
        "effective_restore_norm=",
        pre_summary["effective_restore_norm"],
    )
    print("rank0 direct correct:", rank0_correct)
    print("rank0 min margin:", float(rank0_margins.min().item()))
    for rank in a.restore_ranks:
        r = per_rank[str(rank)]
        print(
            f"rank{rank}: scale={r['selected_scale']} direct_correct={r['direct_correct_sensitive_tokens']} "
            f"min_margin={r['minimum_direct_margin']:.6f} utility_mse={r['candidate_utility_selected_logit_mse']:.6f}"
        )
    print("summary:", root / "summary.json")


if __name__ == "__main__":
    main()
