#!/usr/bin/env python3
"""SURE-TOFU Stage 2: low-rank Base restoration in the direct-forget nullspace.

Starts from a frozen successful Stage-1B rank-0 forgetting checkpoint.
Transformer and input embeddings remain frozen.  The script:

1. builds the complete numerical row span B_F of teacher-forced hidden states
   for every answer-token position in the same training-visible direct forget
   QAs;
2. computes the Base-relative LM-head restoration desired on the union of
   training-visible answer-token rows;
3. projects that desired restoration into the orthogonal complement of B_F;
4. takes a fixed rank-R (e.g. 64 or 128) SVD basis inside that complement;
5. applies the largest predeclared restoration scale that preserves every
   Stage-1B direct-forget NLL requirement after BF16/FP16 materialization.

Because LM-head logit action is delta_w @ hidden, a restoration row lying in
B_F's orthogonal complement has zero action on every cached direct-forget
answer context in exact arithmetic.  The final explicit NLL guard handles
finite-precision materialization.  No retain95, paraphrase, same-author
holdout, real-authors, world-facts, PPL, or final metric is used for selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "SURE-TOFU-forget-nullspace-Base-restoration"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Frozen Stage1B rank-0 checkpoint")
    p.add_argument("--reference-model-path", required=True, help="Original Full-TOFU checkpoint")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--forget-requirements-json", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--restore-rank", type=int, default=64)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--constraint-tolerance", type=float, default=1e-3)
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_scales(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("candidate-scales must not be empty")
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in values):
        raise ValueError("candidate restoration scales must lie in [0,1]")
    values = sorted(set(values), reverse=True)
    if 0.0 not in values:
        values.append(0.0)
    return values


def numerical_row_basis(rows: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("numerical row basis requires a non-empty matrix")
    values = rows.float()
    _, singular, vh = torch.linalg.svd(values, full_matrices=False)
    if singular.numel() == 0:
        raise RuntimeError("hidden-state SVD returned no singular values")
    tol = max(values.shape) * torch.finfo(values.dtype).eps * float(singular.max().item())
    rank = int((singular > tol).sum().item())
    if rank <= 0:
        raise RuntimeError("direct-forget hidden states have zero numerical rank")
    basis = vh[:rank].contiguous()
    gram = basis @ basis.T
    eye = torch.eye(rank, dtype=gram.dtype, device=gram.device)
    orth_error = float((gram - eye).abs().max().item())
    return basis, {
        "rows": int(values.shape[0]),
        "hidden_size": int(values.shape[1]),
        "numerical_rank": rank,
        "svd_tolerance": tol,
        "largest_singular_value": float(singular.max().item()),
        "smallest_retained_singular_value": float(singular[rank - 1].item()),
        "orthogonality_max_error": orth_error,
    }


def project_away(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return rows
    return rows - (rows @ basis.T) @ basis


def build_restore_basis(
    desired_rows: torch.Tensor,
    forget_basis: torch.Tensor,
    requested_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if requested_rank <= 0:
        raise ValueError("restore-rank must be positive")
    null_desired = project_away(desired_rows.float(), forget_basis.float())
    _, singular, vh = torch.linalg.svd(null_desired, full_matrices=False)
    if singular.numel() == 0:
        raise RuntimeError("Base-restoration SVD returned no singular values")
    tol = max(null_desired.shape) * torch.finfo(torch.float32).eps * float(singular.max().item())
    numerical_rank = int((singular > tol).sum().item())
    actual_rank = min(requested_rank, numerical_rank)
    if actual_rank <= 0:
        raise RuntimeError("no Base-restoration direction survives the direct-forget nullspace")

    # Reproject then QR to remove SVD/rounding leakage into the forget span.
    raw_basis = project_away(vh[:actual_rank].float(), forget_basis.float())
    q, _ = torch.linalg.qr(raw_basis.T, mode="reduced")
    restore_basis = q.T.contiguous()
    restore_basis = project_away(restore_basis, forget_basis.float())
    q2, _ = torch.linalg.qr(restore_basis.T, mode="reduced")
    restore_basis = q2.T.contiguous()

    overlap = restore_basis @ forget_basis.float().T
    gram = restore_basis @ restore_basis.T
    eye = torch.eye(actual_rank, dtype=gram.dtype, device=gram.device)
    overlap_max = float(overlap.abs().max().item()) if overlap.numel() else 0.0
    orth_max = float((gram - eye).abs().max().item()) if gram.numel() else 0.0

    # Best Frobenius projection of the Base-relative desired row changes into
    # the fixed rank-R restoration subspace.
    delta = (desired_rows.float() @ restore_basis.T) @ restore_basis
    report = {
        "requested_rank": requested_rank,
        "actual_rank": actual_rank,
        "null_desired_numerical_rank": numerical_rank,
        "svd_tolerance": tol,
        "forget_overlap_max_abs": overlap_max,
        "restore_basis_orthogonality_max_error": orth_max,
        "null_desired_fro_norm": float(null_desired.norm().item()),
        "projected_restore_delta_fro_norm": float(delta.norm().item()),
    }
    return restore_basis, delta, report


def load_required_nll(
    path: str | None,
    count: int,
    fallback_nll: float,
    device: torch.device,
) -> torch.Tensor:
    if path is None:
        return torch.full((count,), fallback_nll, dtype=torch.float32, device=device)
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("forget-requirements-json must contain one row per direct forget QA")
    values: List[float] = []
    for row in rows:
        value = row.get("required_nll", fallback_nll)
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32, device=device)


@torch.no_grad()
def load_reference_rows(
    model_path: str,
    selected_ids: Sequence[int],
    dtype_name: str,
) -> torch.Tensor:
    dtype = gagd.torch_dtype(dtype_name)
    # Keep the teacher on CPU; only the selected rows are moved to the active GPU.
    reference = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    weight = reference.get_output_embeddings().weight
    ids = torch.tensor(selected_ids, dtype=torch.long, device=weight.device)
    rows = weight.index_select(0, ids).detach().float().cpu().contiguous()
    del reference
    gc.collect()
    return rows


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.restore_rank <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("forget-num, restore-rank, batch-size and max-length must be positive")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.constraint_tolerance < 0:
        raise ValueError("constraint-tolerance must be non-negative")
    scales = parse_scales(a.candidate_scales)

    forget_path = Path(a.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    tok_data = AutoTokenizer.from_pretrained(a.model_path)
    if tok_data.pad_token is None:
        tok_data.pad_token = tok_data.eos_token
    instances, source_indices = locked.load_forget_instances(forget_path, tok_data, a.forget_num)

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = input_weight.data_ptr()
    input_version = input_weight._version
    device = gagd.first_device(model)

    baseline_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    fallback_required = -math.log(a.target_forget_answer_probability)
    required_nll = load_required_nll(
        a.forget_requirements_json,
        len(instances),
        fallback_required,
        device,
    )
    baseline_slack = baseline_nll - required_nll
    if bool((baseline_slack < -a.constraint_tolerance).any().item()):
        worst = float(baseline_slack.min().item())
        raise RuntimeError(
            f"Stage1B checkpoint is not forget-feasible before restoration; min slack={worst}"
        )

    all_positions = list(range(len(instances)))
    selected_ids = locked.answer_rows_for_instances(
        tok, instances, all_positions, max_length=a.max_length
    )
    if not selected_ids:
        raise RuntimeError("direct forget answers produced no selected LM-head rows")
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    current_rows = output_weight.index_select(0, selected_tensor).detach().float().clone()

    caches = tofu.build_answer_delta_caches(
        model,
        tok,
        instances,
        selected_ids,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    )
    hidden = torch.cat([cache.hidden for cache in caches], dim=0).float()
    forget_basis, forget_geometry = numerical_row_basis(hidden)
    complement_dim = int(hidden.shape[1] - forget_basis.shape[0])
    if complement_dim <= 0:
        raise RuntimeError("direct-forget hidden states span the entire hidden dimension")

    reference_rows_cpu = load_reference_rows(a.reference_model_path, selected_ids, a.dtype)
    reference_rows = reference_rows_cpu.to(device=current_rows.device)
    desired = reference_rows - current_rows
    restore_basis, restore_delta, restore_geometry = build_restore_basis(
        desired, forget_basis, a.restore_rank
    )

    hidden_action = hidden @ restore_delta.T
    max_hidden_action_fp32 = float(hidden_action.abs().max().item()) if hidden_action.numel() else 0.0
    distance_before = float(desired.norm().item())

    chosen_scale = None
    chosen_nll = None
    scale_trials: List[Dict[str, Any]] = []
    original_materialized_rows = output_weight.index_select(0, selected_tensor).detach().clone()
    for scale in scales:
        candidate_rows = current_rows + float(scale) * restore_delta
        with torch.no_grad():
            output_weight.index_copy_(
                0,
                selected_tensor,
                candidate_rows.to(device=output_weight.device, dtype=output_weight.dtype),
            )
        candidate_nll = tofu.score_answer_instances(
            model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
        ).detach().float()
        slack = candidate_nll - required_nll
        feasible = bool(torch.all(slack >= -a.constraint_tolerance).item())
        scale_trials.append(
            {
                "scale": float(scale),
                "feasible": feasible,
                "minimum_forget_nll_slack": float(slack.min().item()),
                "forget_answer_probability_mean": float(torch.exp(-candidate_nll).mean().item()),
                "forget_answer_probability_max": float(torch.exp(-candidate_nll).max().item()),
            }
        )
        if feasible:
            chosen_scale = float(scale)
            chosen_nll = candidate_nll
            break

    if chosen_scale is None or chosen_nll is None:
        with torch.no_grad():
            output_weight.index_copy_(0, selected_tensor, original_materialized_rows)
        raise RuntimeError("no predeclared restoration scale preserved all direct forget constraints")

    if input_weight.data_ptr() != input_pointer or input_weight._version != input_version:
        raise RuntimeError("SURE-TOFU Stage2 modified input embeddings")

    final_rows = output_weight.index_select(0, selected_tensor).detach().float()
    distance_after = float((reference_rows - final_rows).norm().item())
    final_slack = chosen_nll - required_nll

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    report = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": a.seed,
        "restore_rank_requested": a.restore_rank,
        "restore_rank_actual": restore_geometry["actual_rank"],
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "forget_hidden_geometry": forget_geometry,
        "forget_nullspace_dimension": complement_dim,
        "restore_geometry": restore_geometry,
        "maximum_fp32_direct_forget_logit_action": max_hidden_action_fp32,
        "candidate_scales": scales,
        "scale_trials": scale_trials,
        "chosen_scale": chosen_scale,
        "base_row_distance_before": distance_before,
        "base_row_distance_after": distance_after,
        "base_row_distance_restored_fraction": (
            1.0 - distance_after / distance_before if distance_before > 0 else 0.0
        ),
        "minimum_final_forget_nll_slack": float(final_slack.min().item()),
        "final_forget_answer_probability_mean": float(torch.exp(-chosen_nll).mean().item()),
        "final_forget_answer_probability_max": float(torch.exp(-chosen_nll).max().item()),
        "constraint_source": (
            str(Path(a.forget_requirements_json).resolve())
            if a.forget_requirements_json
            else f"target_probability={a.target_forget_answer_probability}"
        ),
        "training_selection_data_access": {
            "direct_forget_qas": a.forget_num,
            "full_tofu_reference_rows": True,
            "retain95": 0,
            "paraphrases": 0,
            "same_author_holdout": 0,
            "real_authors": 0,
            "world_facts": 0,
            "PPL": False,
        },
        "visible_source_indices": source_indices,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "restoration_report.json", report)
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "model_path": a.model_path,
            "reference_model_path": a.reference_model_path,
            "forget_json": str(forget_path),
            "forget_requirements_json": a.forget_requirements_json,
            "seed": a.seed,
            "forget_num": a.forget_num,
            "restore_rank": a.restore_rank,
            "candidate_scales": scales,
            "chosen_scale": chosen_scale,
            "constraint_tolerance": a.constraint_tolerance,
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print(
        f"SURE-TOFU Stage2 PASS rank={a.restore_rank} actual={restore_geometry['actual_rank']} "
        f"scale={chosen_scale} forget_span_rank={forget_basis.shape[0]} "
        f"null_dim={complement_dim} restored={report['base_row_distance_restored_fraction']:.4f}"
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
